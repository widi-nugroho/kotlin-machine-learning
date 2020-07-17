package knn
class KnnRegressor (val k:Int,val inputweight:List<Double>){
    fun getKNearestNeighbour(sorteddata:List<Pair<Double,DataRegressor>>):List<Pair<Double,DataRegressor>>{
        var res= mutableListOf<Pair<Double,DataRegressor>>()
        for (i in 0..k-1){
            res.add(sorteddata[i])
        }
        return res
    }
    fun computeDistance(input:DataRegressor, data:List<DataRegressor>):List<Pair<Double,DataRegressor>>{
        var res= mutableListOf<Pair<Double,DataRegressor>>()
        for (i in data){
            var dist=Distance.weightedEuclidDistance(inputweight,input,i)
            res.add(Pair(dist,i))
        }
        return res.sortedWith(compareBy({it.first}))
    }
    fun kNearestNeighbourMeanOutput(sorted:List<Pair<Double,DataRegressor>>):Double{
        var res=0.0
        for (i in sorted){
            res+=i.second.output
        }
        return res/sorted.size.toDouble()
    }
    fun regress(input:List<Double>,data:List<DataRegressor>):Double{
        var c=DataRegressor(input,0.0)
        var cd=computeDistance(c,data)
        var cd2=getKNearestNeighbour(cd)
        var md=kNearestNeighbourMeanOutput(cd2)
        return md
    }
}
fun main(){
    var weight= listOf<Double>(0.001,1.0,10.0,0.1,1000.0)
    var test= listOf<Double>(1000.0,0.0,0.3048,71.3,0.00266337)
    var r=DataAccess.loadAirFoilNoise("/home/widi/projects/kotlin-machine-learning/src/main/resources/airfoil_self_noise.csv")
    var k2=KnnRegressor(3,weight)
    println(k2.regress(test,r))
}
