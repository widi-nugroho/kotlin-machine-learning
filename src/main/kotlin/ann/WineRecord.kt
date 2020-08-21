package ann

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDArrays
import ai.djl.ndarray.NDManager

class WineRecord(){
    var input= mutableListOf<Double>()
    var output= mutableListOf<Double>()
    constructor(i:MutableList<Double>,o:Int):this(){
        this.input=i
        if (o==1){
            this.output= mutableListOf(1.0,0.0,0.0)
        }else if(o==2){
            this.output= mutableListOf(0.0,1.0,0.0)
        }else{
            this.output= mutableListOf(0.0,0.0,1.0)
        }
    }
}