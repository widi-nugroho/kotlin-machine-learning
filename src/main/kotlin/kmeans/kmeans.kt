package kmeans

import kotlin.math.sqrt
import kotlin.random.Random
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
}



fun main(){
    var irisRecord=RecordAccess.loadcsv("/home/widi/projects/kotlin-machine-learning/src/main/resources/iris.data")
    var iris=Kmeans(3)
    iris.initCentroids(irisRecord)
    for(i in iris.centroids){
        println(i.features)
    }
    iris.train(0.2,100,irisRecord)
    for(i in iris.centroids){
        println(i.features)
    }

}