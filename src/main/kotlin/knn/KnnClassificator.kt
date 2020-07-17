package knn

class KnnClassificator(val k:Int) {
    fun getKNearestNeighbour(sorteddata:List<Pair<Double,DataClassificator>>):List<Pair<Double,DataClassificator>>{
        var res= mutableListOf<Pair<Double,DataClassificator>>()
        for (i in 0..k-1){
            res.add(sorteddata[i])
        }
        return res
    }
    fun computeDistance(input:DataClassificator, data:List<DataClassificator>):List<Pair<Double,DataClassificator>>{
        var res= mutableListOf<Pair<Double,DataClassificator>>()
        for (i in data){
            var dist=Distance.euclidDistanceClassificator(i,input)
            res.add(Pair(dist,i))
        }
        return res.sortedWith(compareBy({it.first}))
    }
    fun kNearestNeighbourGroupedByLabel(sorted:List<Pair<Double,DataClassificator>>):List<Pair<String,Int>>{
        var group= mutableMapOf<String,Int>()
        for (i in sorted){
            if (group[i.second.output]==null){
                group[i.second.output]=1
            }else{
                group[i.second.output]=group[i.second.output]!!+1
            }
        }
        var res= mutableListOf<Pair<String,Int>>()
        for ((k,v) in group){
            res.add(Pair(k,v))
        }
        return res.sortedWith(compareByDescending { it.second })
    }
    fun classify(input:List<Double>,data:List<DataClassificator>):String{
        var c=DataClassificator(input," ")
        var cd=computeDistance(c,data)
        var cd2=getKNearestNeighbour(cd)
        var md=kNearestNeighbourGroupedByLabel(cd)
        return md[0].first
    }
}
fun main(){
    var t=DataAccess.loadcsv("/home/widi/projects/kotlin-machine-learning/src/main/resources/iris.data")
    var k1=KnnClassificator(100)
    var t2=k1.getKNearestNeighbour(k1.computeDistance(t[0],t))
    println(k1.kNearestNeighbourGroupedByLabel(t2))
    //for (i in 0..t.size-1){
    //    println(t[0].output+"dist: "+k1.computeDistance(t[0],t)[i].first+"datanya: "+k1.computeDistance(t[0],t)[i].second.output)
    //}
    var test= listOf<Double>(4.9,3.0,1.4,0.2)
    println(k1.classify(test,t))
}