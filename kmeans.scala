
import scala.io.Source
import java.io.{FileReader, FileNotFoundException, IOException}
import scala.math._
import scala.util.Random
import scala.collection.mutable.ListBuffer

object kmeans extends App {
/* Arguments for the algorithm */
val nClusters = 2
val nIters = 5
val filename = "kmSample1.txt"

/* Read the file containing the cluster data */
val designMatrix = Source.fromFile(filename).getLines().map(_.split(",").toList.map(_.trim.toDouble)).toList

/* Generate random centers for nClusters */
val randomIndices = Random.shuffle(0 to designMatrix.size).take(nClusters)
val randomInitCenters = randomIndices.toList collect designMatrix
var prevClusterCenters = randomInitCenters

/* Main part to run kmeans algorithm for nIters */
for (i <- 0 until nIters) {
 val assignedClusterIdx = assignToCluster(designMatrix,prevClusterCenters)
 val newClusterCenters = calcClusterCenters(designMatrix,assignedClusterIdx)
 prevClusterCenters = newClusterCenters
 println("Iteration: " )
  println("Cluster Centers: " + newClusterCenters)
}

/* Define Functions */

/* Calculate Euclidean distance for between two nFeature sized points:nFeatures (= number of columns in file) */
def distanceXY(fromThis:List[Double],toThat:List[Double]) ={
sqrt((fromThis, toThat).zipped.map(_ - _).map(pow(_,2)).sum) }

/* Calculate sum of nFeature sized points:nFeatures (= number of columns in file) */
def sumXY(fromThis:List[Double],toThat:List[Double]) ={
(fromThis, toThat).zipped.map(_ + _) }

/* Assign points to closest cluster center as a collection*/ 
def assignToCluster(X:List[List[Double]],y:List[List[Double]]):List[Int] = {
var assignedClusterIdx = new ListBuffer[Int]()
for (i <- 0 until X.size) {
var x = new ListBuffer[Double]()
  for (j <- 0 until y.size) {
     x += distanceXY(X(i),y(j))
    }
  assignedClusterIdx += x.zipWithIndex.min._2
 }
 return assignedClusterIdx.toList
}

/* Recalculate cluster centers as mean of its points collection */
def calcClusterCenters (X:List[List[Double]], idx:List[Int]): List[List[Double]] = {
val groupByCluster = X zip idx groupBy(_._2) transform ((i:Int, p:List[(List[Double], Int)]) => for (x <- p) yield x._1)

var clusterCenters = new ListBuffer[List[Double]]()
for ((k,v) <- groupByCluster) {
var clusterSum:List[Double] = List.fill(X(0).size)(0.0)
  for (i <- 0 until v.size) {  clusterSum = sumXY(clusterSum,v(i))}
  clusterCenters += clusterSum.map(_/v.size)
   }
val centroids:List[List[Double]] = clusterCenters.toList
return centroids
  }
 }
