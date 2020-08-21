package ann

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDManager
import knn.DataAccess

class WineRecord(){
    var input= mutableListOf<Float>()
    var output= mutableListOf<Float>()
    constructor(i:MutableList<Float>,o:Int):this(){
        this.input=i
        if (o==1){
            this.output= mutableListOf(1.0f,0.0f,0.0f)
        }else if(o==2){
            this.output= mutableListOf(0.0f,1.0f,0.0f)
        }else{
            this.output= mutableListOf(0.0f,0.0f,1.0f)
        }
    }
}
fun main(){
    var manager=NDManager.newBaseManager()
    var wineDataset= DataAccess.WineList("/home/widi/projects/kotlin-machine-learning/src/main/resources/wine.data")
    var p=split(wineDataset,manager)
    println(p.first[0])
}