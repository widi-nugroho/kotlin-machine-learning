package kmeans

import org.math.plot.Plot2DPanel
import java.awt.Color
import kotlin.math.sqrt
import kotlin.random.Random
import javax.swing.JFrame

class Kmeans(){
    var k:Int=0
    constructor(k:Int):this(){
        this.k=k
    }
    var centroids= mutableListOf<Centroid>()
    fun initCentroids(records:List<Record>){
        var res= mutableListOf<Centroid>()
        for (i in 0..k-1){
            var r=records[Random.nextInt(0,records.size-1)]
            var r2=Centroid(r.features,r.output)
            centroids.add(r2)
        }
    }
    fun euclidDistance(a:Record,b:Record):Double{
        var res=0.0
        for ((k,v) in a.features){
            var selisih=a.features[k]!!-b.features[k]!!
            selisih=selisih*selisih
            res+=selisih
        }
        return sqrt(res)
    }
    fun findWinner(a:Record):Int{
        var dist= mutableListOf<Pair<Int,Double>>()
        for (i in 0..centroids.size-1){
            dist.add(Pair(i, euclidDistance(a,centroids[i])))
        }
        var sortedDist=dist.sortedWith(compareBy{it.second})
        return sortedDist[0].first
    }
    fun assignToCentroid(a:List<Record>){
        for (i in a){
            var t= findWinner(i)
            centroids[t].members.add(i)
        }
    }
    fun updateCentroids(lr:Double){
        for (i in centroids){
            i.updateCentroid(lr)
        }
    }
    fun clearMember(){
        for (i in centroids){
            i.clearRecord()
        }
    }
    fun train(lr: Double,n:Int,trainingData:List<Record>){
        for (i in 0..n-1){
            clearMember()
            assignToCentroid(trainingData)
            updateCentroids(lr)
        }
    }
    fun plotting(datas:List<Record>){
        var plot= Plot2DPanel()
        val frame = JFrame("Kmeans Iris Plot")
        frame.setSize(600, 600)
        var xd= mutableListOf<Double>()
        var yd= mutableListOf<Double>()
        var xc= mutableListOf<Double>()
        var yc= mutableListOf<Double>()
        for (i in datas){
            var(x1,y1)=i.coordinate()
            xd.add(x1)
            yd.add(y1)
        }
        for (i in centroids){
            var(x1,y1)=i.coordinate()
            xc.add(x1)
            yc.add(y1)
        }
        plot.addScatterPlot("Datas plot", Color.BLACK,xd.toDoubleArray(),yd.toDoubleArray())
        plot.addScatterPlot("Centroids plot", Color.RED,xc.toDoubleArray(),yc.toDoubleArray())
        frame.contentPane=plot
        frame.isVisible=true
    }
    fun min_distance_between_centroids(): Double {
        var res= mutableListOf<Double>()
        for (i in 0..centroids.size-1){
            for (j in i..centroids.size-1){
                if (i!=j){
                    var d=euclidDistance(centroids[i],centroids[j])
                    res.add(d)
                }
            }
        }
        return res.sorted()[0]
    }
    fun max_distance_centroids(): Double {
        var res= mutableListOf<Double>()
        for (i in centroids){
            res.add(i.max_distance_to_centroids())
        }
        return res.sorted()[res.lastIndex]
    }
    fun dunn_index(): Double {
        var max=max_distance_centroids()
        var min=min_distance_between_centroids()
        return min/max
    }
}



fun main(){
    var irisRecord=RecordAccess.loadcsv("/home/widi/projects/kotlin-machine-learning/src/main/resources/iris.data")
    var iris=Kmeans(3)
    iris.initCentroids(irisRecord)
    for(i in iris.centroids){
        println(i.features)
    }
    iris.plotting(irisRecord)
    iris.train(0.2,100,irisRecord)
    for(i in iris.centroids){
        println(i.features)
    }
    iris.plotting(irisRecord)
    /*iris.min_distance_between_centroids()
    var dis=iris.min_distance_between_centroids()
    println("         "+dis)
    iris.centroids[0].max_distance_to_centroids()
    var t=iris.centroids[0].max_distance_to_centroids()
    println("              "+t)*/
    println(iris.dunn_index())
}