package kmeans

import kotlin.math.sqrt

open class Record {
    var output=""
    var features= mutableMapOf<String,Double>()
    constructor(features:MutableMap<String,Double>,output:String){
        this.features=features
        this.output=output
    }
    fun coordinate(): Pair<Double, Double> {
        var x=(features["sepal_L"]!!+features["petal_L"]!!)/2
        var y=(features["sepal_W"]!!+features["petal_W"]!!)/2
        return Pair(x,y)
    }
    fun euclidDistance(b:Record):Double{
        var res=0.0
        for ((k,v) in this.features){
            var selisih=this.features[k]!!-b.features[k]!!
            selisih=selisih*selisih
            res+=selisih
        }
        return sqrt(res)
    }
}

class Centroid :Record {
    var members = mutableListOf<Record>()
    constructor(features: MutableMap<String, Double>, output: String):super(features, output) {

    }
    fun addRecord(a:Record){
        this.members.add(a)
    }
    fun clearRecord(){
        this.members.removeAll(members)
    }
    fun updateCentroid(lr:Double){
        var avg=computeMean()
        for ((k,v)in avg){
            this.features[k]=(1-lr)*this.features[k]!!+lr*v!!
        }
    }
    fun computeMean():MutableMap<String,Double>{
        var res= mutableMapOf<String,Double>()
        var sepal_L= mutableListOf<Double>()
        var sepal_W= mutableListOf<Double>()
        var petal_L= mutableListOf<Double>()
        var petal_W= mutableListOf<Double>()
        for (i in members){
            var sepalL=i.features["sepal_L"]!!
            var sepalW=i.features["sepal_W"]!!
            var petalL=i.features["petal_L"]!!
            var petalW=i.features["petal_W"]!!
            sepal_L.add(sepalL)
            sepal_W.add(sepalW)
            petal_L.add(petalL)
            petal_W.add(petalW)
        }
        var avg1=sepal_L.average()
        var avg2=sepal_W.average()
        var avg3=petal_L.average()
        var avg4=petal_W.average()
        res["sepal_L"]=avg1
        res["sepal_W"]=avg2
        res["petal_L"]=avg3
        res["petal_W"]=avg4
        return res
    }
    fun max_distance_to_centroids(): Double {
        var res= mutableListOf<Double>()
        for (i in members){
            var d=i.euclidDistance(this)
            res.add(d)
        }
        return res.sorted()[res.lastIndex]
    }
}