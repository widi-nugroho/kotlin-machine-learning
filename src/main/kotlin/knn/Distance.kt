package knn

import kotlin.math.sqrt

object Distance {
    fun euclidDistanceClassificator(d1:DataClassificator, d2:DataClassificator):Double{
        var res=0.0
        for (i in 0..d1.input.size-1){
            var selisih=d1.input[i]-d2.input[i]
            var selisih2=selisih*selisih
            res+=selisih2
        }
        return sqrt(res).toDouble()
    }
    fun weightedEuclidDistance(weight:List<Double>,d1:DataRegressor,d2:DataRegressor):Double{
        var res=0.0
        for (i in 0..d1.input.size-1){
            var selisih=weight[i]*(d1.input[i]-d2.input[i])
            var selisih2=selisih*selisih
            res+=selisih2
        }
        return sqrt(res).toDouble()
    }
}

fun main(){
    var t=DataAccess.loadcsv("/home/widi/projects/kotlin-machine-learning/src/main/resources/iris.data")
    println(Distance.euclidDistanceClassificator(t[0],t[1]))
}