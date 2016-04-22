

import scala.util.Random
import org.apache.spark.{SparkConf, SparkContext}

import scala.reflect.ClassTag
import breeze.linalg.{DenseMatrix, DenseVector}
import com.quantifind.charts.Highcharts.{
  delete => del, hold, legend => leg, line, redo => red, setPort, startServer, stopServer,
  title => tit, undo => und, unhold, xAxis => xAx, yAxis => yAx
}

import scala.io.StdIn
import org.apache.spark.SparkContext

 /*
 *
 * This example shows how to generate sample data to run linear regression using
 * both Scala collections and Spark. The optimizers used here are stochastic
 * gradient descent (SGD) and Adagrad.
 *
 * @author Marek Kolodziej
 */
object SGDAdagradDemo extends App {

  /*@transient val log = Logger.getLogger(SGDAdagradDemo.getClass)*/
  
    // Set up Spark
  val conf = new SparkConf().setAppName("sgd_and_adagrad_demo").setMaster("local[1]")
  val sc = new SparkContext(conf)

  case class Datum(target: Double, features: DenseVector[Double]) {
  override def toString: String =
    s"Datum(target = $target, features = $features)"
}
 
   case class GradFn(f: (DistData[VectorizedData], DenseVector[Double]) => DenseVector[Double]) {
    def apply(data: DistData[VectorizedData], weights: DenseVector[Double]) =
      f(data, weights)
  }

  case class CostFn(f: (DistData[VectorizedData], DenseVector[Double]) => Double) {
    def apply(data: DistData[VectorizedData], weights: DenseVector[Double]) =
      f(data, weights)
  }

  case class OptHistory(
                       cost: Seq[Double],
                       weights: Seq[DenseVector[Double]],
                       grads: Seq[DenseVector[Double]]
                     )
                     
  case class WeightUpdate(f: (DistData[VectorizedData], OptHistory, GradFn, CostFn,
    Double, Double, Double, Int, Long) => OptHistory) {
    def apply(
               data: DistData[VectorizedData],
               history: OptHistory,
               gradFn: GradFn,
               costFn: CostFn,
               initAlpha: Double,
               momentum: Double,
               miniBatchFraction: Double,
               miniBatchIterNum: Int,
               seed: Long
               ): OptHistory =
      f(
        data, history, gradFn, costFn, initAlpha,
        momentum, miniBatchFraction, miniBatchIterNum, seed
      )
  }

  case class WeightInit(f: (Int, Long) => DenseVector[Double]) {
    def apply(numEl: Int, seed: Long): DenseVector[Double] =
      f(numEl, seed)
  }
  
/**
 * Trait that abstractly represents operations that can be performed on a dataset.
 * The implementation of DistData is suitable for both large-scale, distributed data
 * or in-memory structures.
 *
 * @author Malcolm Greaves
 * @author Marek Kolodziej
 */
trait DistData[A] {

  /* Transform a dataset by applying f to each element. */
  def map[B: ClassTag](f: A => B): DistData[B]

  /* This has type A as opposed to B >: A due to the RDD limitations */
  def reduceLeft(op: (A, A) => A): A

  /*
   * Starting from a defined zero value, perform an operation seqOp on each element
   * of a dataset. Combine results of seqOp using combOp for a final value.
   */
  def aggregate[B: ClassTag](zero: B)(seqOp: (B, A) => B, combOp: (B, B) => B): B

  /** Sort the dataset using a function f that evaluates each element to an orderable type */
  def sortBy[B: ClassTag](f: (A) ⇒ B)(implicit ord: math.Ordering[B]): DistData[A]

  /** Construct a traversable for the first k elements of a dataset. Will load into main mem. */
  def take(k: Int): Traversable[A]

  /** Load all elements of the dataset into an array in main memory. */
  def toSeq: Seq[A]

  def flatMap[B: ClassTag](f: A => TraversableOnce[B]): DistData[B]

  def groupBy[B: ClassTag](f: A => B): DistData[(B, Iterable[A])]

  def sample(withReplacement: Boolean, fraction: Double, seed: Long): DistData[A]

  def exactSample(fraction: Double, seed: Long): DistData[A]

  def size: Long

  def headOption: Option[A]

  def zipWithIndex: DistData[(A, Long)]

  def foreach(f: A => Unit): Unit
}
object Sampling {
/*
   * Helper for sampling with and without replacement of Scala collections.
   *
   * @param coll
   * @param sampleSize
   * @param withReplacement
   * @param seed
   * @tparam A
   * @return
   */
  def sample[A](coll: Traversable[A], sampleSize: Int,
                withReplacement: Boolean, seed: Long = System.nanoTime): IndexedSeq[A] = {

    val indexed = coll.toIndexedSeq

    val rand = new Random(seed)

    /* Tail-recursive helper for sampling without replacement.
       Add picked element to acc and remove it from seq so
       it can't be chosen again.
     */
    @annotation.tailrec
    def collect(seq: IndexedSeq[A], size: Int, acc: List[A]): List[A] = {
      if (size == 0) acc
      else {
        val index = rand.nextInt(seq.size)
        collect(seq.updated(index, seq(0)).tail, size - 1, seq(index) :: acc)
      }
    }

    // simple sampling with replacement
    def withRep: IndexedSeq[A] =
      for (i <- 1 to sampleSize)
        yield indexed(rand.nextInt(coll.size))

    if (withReplacement)
      withRep
    else
      collect(indexed, sampleSize, Nil).toIndexedSeq
  }
    /*
   * Helper to sample for mini-batches. If the numer of items in the DistData instance
   * (Scala collection or RDD) is greater than or equal to 1/miniBatchFraction, use
   * DistData's own sampling (note that we're using exactSampleWithoutReplacment() as
   * opposed to sample, because Spark has a bug in its sample() method. When it is fixed,
   * this will default back to calling sample(). When the number of items in the
   * DistData instance is less than 1/miniBatchFraction, we need to sample directly from the
   * VectorizedData instances (rows within the matrices).
   *
   * Example: if we have 100 elements in DistData (100 instances of VectorizedData),
   * then sampling 0.01 (from 1/0.01 = 100 elements) means choosing 1 element at random.
   * However, if we have 10 VectorizedData instances in a DistData instance, then sampling
   * 1% is impossible without sampling 1% directly from each VectorizedData element
   * within DistData. The latter is slower, but compensates for too much aggregation
   * of individual examples into VectorizedData instances.
   *
   * @param data
   * @param miniBatchFraction
   * @param currSeed
   * @return
   */
  def sampleMiniBatch(data: DistData[VectorizedData],
                      miniBatchFraction: Double,
                      currSeed: Long): DistData[VectorizedData] = {

    val collCount = data.size

    val regularSampling = collCount >= math.ceil(1.0 / miniBatchFraction)

    if (regularSampling) {

      data.exactSample(fraction = miniBatchFraction, seed = currSeed)

    } else {

      data.map {

        case v: VectorizedData =>

          val size = v.target.activeSize
          val rounded = math.max(1, math.round(miniBatchFraction * size).toInt)

          val rowIndices = sample(
            coll = (0 until size),
            sampleSize = rounded,
            withReplacement = false,
            seed = currSeed
          )

          VectorizedData(
            target = v.target(rowIndices).toDenseVector,
            features = v.features(rowIndices, ::).toDenseMatrix
          )
      }
    }
  }
}

/** Wraps a Traversable as a DistData. */
case class TravDistData[A: ClassTag](ls: Traversable[A]) extends DistData[A] {

  override def map[B: ClassTag](f: A => B): DistData[B] =
    new TravDistData(ls.map(f))

  override def reduceLeft(op: (A, A) => A): A =
    ls.reduceLeft(op)

  override def aggregate[B: ClassTag](zero: B)(seqOp: (B, A) => B, combOp: (B, B) => B): B =
    ls.aggregate(zero)(seqOp, combOp)

  override def sortBy[B: ClassTag](f: (A) ⇒ B)(implicit ord: math.Ordering[B]): DistData[A] =
    new TravDistData(ls.toSeq.sortBy(f))

  override def take(k: Int): Traversable[A] =
    ls.take(k)

  override def toSeq: Seq[A] =
    ls.toSeq

  override def flatMap[B: ClassTag](f: A => TraversableOnce[B]): DistData[B] =
    new TravDistData(ls.flatMap(f))

  override def groupBy[B: ClassTag](f: A => B): DistData[(B, Iterable[A])] =
    new TravDistData(ls.groupBy(f).toTraversable.map({ case (b, iter) => (b, iter.toIterable) }))

  override def sample(withReplacement: Boolean, fraction: Double, seed: Long): DistData[A] =
    Sampling.sample(ls, math.round(fraction * ls.size).toInt, withReplacement, seed)

  override def exactSample(fraction: Double, seed: Long): DistData[A] =
    sample(withReplacement = false, fraction = fraction, seed = seed)

  override def size: Long =
    ls.size

  override def headOption: Option[A] =
    ls.headOption

  override def zipWithIndex: DistData[(A, Long)] =
    ls.toIndexedSeq.zipWithIndex.map(a => (a._1, a._2.toLong))

  override def foreach(f: A => Unit): Unit =
    ls.foreach(f)
}
object DistData {
  /** Implicitly converts a Traversable into a DistData type. */
  @inline implicit def traversable2DistData[A: ClassTag](l: Traversable[A]): DistData[A] =
    TravDistData(l)
}
/** Type that allows us to convert an interable sequence of data into a DistData type. */

trait DistDataContext {
  def from[T: ClassTag](data: Iterable[T]): DistData[T]
}

object DistDataContext {

  /** Implicitly converts a SparkContext into a DistDataContext type. */

  implicit val travDDContext: DistDataContext =
    TraversableDistDataContext
}


case object TraversableDistDataContext extends DistDataContext {

  override def from[T: ClassTag](data: Iterable[T]): DistData[T] =
    data.toSeq
}

/*
 * Group a bunch of examples into a feature matrix and a target vector,
 * instead of processing a feature vector and a target value at a time.
 * This will allow for vectorizing the linear algebra. When Breeze's
 * BLAS support is available (see https://github.com/fommil/netlib-java),
 * Breeze will execute linear algebra operations natively, benefiting from
 * lack of garbage collection, vectorization via SSE, etc.
 *
 * @author Marek Kolodziej
 *
 * @param target
 * @param features
 */
case class VectorizedData(target: DenseVector[Double], features: DenseMatrix[Double]) {
  override def toString: String =
    s"""VectorizedData(
       |target = $target,
       |features = $features
       |)
       """.stripMargin
}
/*object DataConversions {


  
   * Convert a DistData instance containing individual Datum records to
   * DistData containing records grouped into VectorizedData for
   * vectorized linear algebra.
   *
   * @param data
   * @param numExamplesPerGroup
   * @return
   */
  def toVectorizedData(data: DistData[Datum], numExamplesPerGroup: Int): DistData[VectorizedData] = {

    val exampleCount = data.size
    val numGroups = exampleCount / numExamplesPerGroup

    val grouped = data.
      zipWithIndex.
      groupBy { case (_, idx) => idx % numGroups }.
      map {
        case (idx, iter) =>
          iter.map {
            case (datum, _) => datum
          }
      }

    grouped.
      map {
      case arr =>
        val numFeat = arr.headOption match {
          case Some(x) => x.features.iterableSize
          case None => 0
        }
        val init = VectorizedData(
          target = DenseVector.zeros[Double](0),
          features = DenseMatrix.zeros[Double](0, numFeat)
        )
        arr.foldLeft(init)(
          (acc, elem) => {
            val vecCat = DenseVector.vertcat(acc.target, DenseVector(elem.target))
            val featMat = elem.features.toDenseMatrix
            val matCat = DenseMatrix.vertcat(acc.features, featMat)
            VectorizedData(vecCat, matCat)
          }
        )
    }
  }
//}

  def optimize(
                iter: Int,
                seed: Long = 42L,
                initAlpha: Double = 0.1,
                momentum: Double = 0.0,
                gradFn: GradFn,
                costFn: CostFn,
                updateFn: WeightUpdate,
                miniBatchFraction: Double,
                weightInitializer: WeightInit,
                data: DistData[VectorizedData]
                ): OptHistory = {

    val count = data.size
    val dataSize = data.headOption match {
      case Some(x) => x.features.cols
      case None => 0
    }
    val exampleCount = data.map(i => i.target.activeSize).reduceLeft(_ + _)
    val initWts = weightInitializer(dataSize, seed)
    val initGrads = weightInitializer(dataSize, seed + 1)
    val initCost = costFn(data, initWts) / (miniBatchFraction * exampleCount)
    // we need 2 steps of history at initialization time for momentum to work correctly
    val initHistory = OptHistory(cost = Seq(initCost, initCost), weights = Seq(initWts, initWts), grads = Seq(initGrads, initGrads))

    (1 to iter).foldLeft(initHistory) {

      case (history, it) =>

        if (it == iter)
          history
        else
          updateFn(data, history, gradFn, costFn, initAlpha, momentum, miniBatchFraction, it, seed)
    }
  }
  
  
  // Generate data for linear regression: y = 3.0 + 10.0 * x + error
  val rand = new Random(42L)
  val numExamples = 1000
  val (intercept, slope) = (3.0D, 10.0D)
  val feature = Seq.fill(numExamples)(rand.nextDouble())
  val targets = feature.map(i => intercept + slope * i + rand.nextDouble() / 100)
  val data =
    targets.
      zip(feature).
      map {
        // merge target and feature, add intercept to feature vector
        case (y, x) => Datum(y, DenseVector[Double](1.0D, x))
      }

  val allFrac = 0.1
  val numGroups = 200
  val allIt = 200

  val localData: DistData[VectorizedData] = toVectorizedData(data = data, numExamplesPerGroup = 10)
  val rdd: DistData[VectorizedData] = toVectorizedData(data = sc.parallelize(data), numExamplesPerGroup = 10)

  /* partially applied function - we'll specify the weight update algorithm and dataset
     for the two cases separately, while reusing common stuff
   */
      val gaussianInit = WeightInit(
    f = (numEl: Int, seed: Long) => {
      val rand = new Random(seed)
      new DenseVector[Double](Array.fill(numEl)(rand.nextGaussian()))
    }
  )
    val linearRegressionCost = CostFn(
    f =
      (data: DistData[VectorizedData], weights: DenseVector[Double]) => {

        val counts = data.map(_.target.activeSize).reduceLeft(_ + _)

        val unscaledCost = data.aggregate(0.0D)(
          seqOp = {

            case (currCost, elem) => {

              currCost + (elem.features * weights :- elem.target).
                map(i => math.pow(i, 2)).
                reduceLeft(_ + _)
            }
          },
          combOp = {

            case (a, b) => a + b
          }
        )

        unscaledCost / (2 * counts)
      }
  )
  val linearRegressionGradient = GradFn(
    f =
      (data: DistData[VectorizedData], weights: DenseVector[Double]) => {
        data.aggregate(DenseVector.zeros[Double](weights.iterableSize))(
          seqOp = {

            case (partialGrad: DenseVector[Double], datum) =>
              datum.features.t * (datum.features * weights :- datum.target)
          },
          combOp = {

            case (partVec1, partVec2) => partVec1 :+ partVec2
          }
        )
      }
  )
  

   private case class OptInfo(
                              private val data: DistData[VectorizedData],
                              private val miniBatchFraction: Double,
                              private val currSeed: Long,
                              private val history: OptHistory,
                              private val costFn: CostFn,
                              private val gradFn: GradFn
                              ) {

    val weights = history.weights.last
    private val histLen = history.cost.size
    lazy val sample = Sampling.sampleMiniBatch(data, miniBatchFraction, currSeed)
    lazy val sampleSize = sample.map(_.target.activeSize).reduceLeft(_ + _)
    lazy val newCost = costFn(sample, weights)
    lazy val gradients = gradFn(sample, weights)
    lazy val prevDeltaW = history.weights(histLen - 1) :- history.weights(histLen - 2)
  }

  /* stochastic gradient descent
     see http://leon.bottou.org/publications/pdf/online-1998.pdf
   */
  val sgd = WeightUpdate(
    f = (data: DistData[VectorizedData],
         history: OptHistory,
         gradFn: GradFn,
         costFn: CostFn,
         initAlpha: Double,
         momentum: Double,
         miniBatchFraction: Double,
         miniBatchIterNum: Int,
         seed: Long) => {

      val opt = OptInfo(data, miniBatchFraction, seed + miniBatchIterNum, history, costFn, gradFn)
      val eta = initAlpha / math.sqrt(opt.sampleSize * miniBatchIterNum)
      val mom: DenseVector[Double] = opt.prevDeltaW :* momentum
      val newWtsNoMom: DenseVector[Double] = opt.weights :- (opt.gradients :* eta)
      val gradWithMom = (opt.gradients :* eta) :+ mom
      val newWtsWithMom = newWtsNoMom :+ mom
      OptHistory(
        cost = history.cost :+ opt.newCost,
        weights = history.weights :+ newWtsWithMom,
        grads = history.grads :+ gradWithMom
      )
    }
  )

  /* Adagrad
     see http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
   */
  val adaGrad = WeightUpdate(
    f = (data: DistData[VectorizedData],
         history: OptHistory,
         gradFn: GradFn,
         costFn: CostFn,
         initAlpha: Double,
         momentum: Double,
         miniBatchFraction: Double,
         miniBatchIterNum: Int,
         seed: Long) => {

      val opt = OptInfo(data, miniBatchFraction, seed + miniBatchIterNum, history, costFn, gradFn)
      val mom: DenseVector[Double] = opt.prevDeltaW :* momentum
      val adaGradDiag: DenseVector[Double] =
        history.grads.foldLeft(
          DenseVector.zeros[Double](opt.weights.iterableSize)
        )(
            (acc: DenseVector[Double], item: DenseVector[Double]) => {
              val temp: Array[Double] = acc.toArray.zip(item.toArray).map(i => i._1 + math.pow(i._2, 2))
              new DenseVector[Double](temp)
            })
      val scaledByDiag = new DenseVector[Double](
        opt.gradients.toArray.zip(adaGradDiag.toArray).map(
          i =>
            initAlpha * i._1 / math.sqrt(i._2)
        )
      )
      val adaGradWts = (opt.weights :- scaledByDiag) :+ mom
      OptHistory(
        cost = history.cost :+ opt.newCost,
        weights = history.weights :+ adaGradWts,
        grads = history.grads :+ scaledByDiag
      )
    }
  )
  
  val commonParams = optimize(
    iter = allIt,
    seed = 123L,
    initAlpha = 0.1,
    momentum = 0.9,
    gradFn = linearRegressionGradient,
    costFn = linearRegressionCost,
    _: WeightUpdate,
    miniBatchFraction = allFrac,
    weightInitializer = gaussianInit,
    _: DistData[VectorizedData]
  )

  val vectLocalOptSGDWithMomentum =
    commonParams(sgd, localData)

  val vectSparkOptAdaWithMomentum =
    commonParams(adaGrad, rdd)

/*
 * WISP wrapper for multiple line plots on the same chart
 *
 * @author Marek Kolodziej
 *
 * @param x
 * @param y
 * @param xAxis
 * @param yAxis
 * @param title
 * @param legend
 * @param port
 */
case class MultiPlot(x: Seq[Int],
                     y: Seq[Seq[Double]],
                     xAxis: String,
                     yAxis: String,
                     title: String,
                     legend: Seq[String],
                     port: Int = 1234
                      ) {

  def close(): Unit =
    stopServer

  def delete(): Unit =
    del()

  def redo(): Unit =
    red

  def render(): Unit = {

    setPort(port)
    startServer
    line((x, y(0)))
    hold()
    y.tail.foreach(
      series =>
        line((x, series))
    )
    xAx(xAxis)
    yAx(yAxis)
    tit(title)
    leg(legend)
    unhold()
  }

  def renderAndCloseOnAnyKey(): Unit = {

    render()
    StdIn.readLine()
    close()
  }

  def undo(): Unit =
    und()


}

  val dropEl = 3
  val plot = MultiPlot(
    x = dropEl to allIt,
    y = Seq(
      vectLocalOptSGDWithMomentum.cost.drop(dropEl),
      vectSparkOptAdaWithMomentum.cost.drop(dropEl)
    ),
    xAxis = "Iteration",
    yAxis = "Cost",
    title = s"Expected: y ≈ $intercept + $slope * x",
    legend = Seq(
      f"Scala collections + SGD: y ≈ ${vectLocalOptSGDWithMomentum.weights.last(0)}%.1f + ${vectLocalOptSGDWithMomentum.weights.last(1)}%.1f * x",
      f"Scala collections + Adagrad: y ≈ ${vectSparkOptAdaWithMomentum.weights.last(0)}%.1f + ${vectSparkOptAdaWithMomentum.weights.last(1)}%.1f * x"
    ),
    port = 1234
  )

  plot.renderAndCloseOnAnyKey()

} 
