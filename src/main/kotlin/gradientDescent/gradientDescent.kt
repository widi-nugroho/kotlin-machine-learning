package gradientDescent

import kotlin.math.abs

class Differential() {
    var h=0.0
    lateinit var f:(Double)->Double
    constructor(f:(Double)->Double,h:Double):this(){
        this.f=f
        this.h=h
    }
    fun computeDifferential(x:Double):Double{
        var output=(this.f(x+this.h)-this.f(x))/this.h
        return output
    }
}
class Integration(){
    var h=0.0
    lateinit var f:(Double)->Double
    constructor(f:(Double)->Double,h: Double):this(){
        this.h=h
        this.f=f
    }
    fun computeIntegration(a:Double,b:Double):Double{
        var rieman=0.0
        var xnew=a
        while (xnew<=b){
            var sekali=this.f(xnew)*this.h
            xnew+=this.h
            rieman+=sekali
        }
        return rieman
    }
}
class GradientDescent(){
    var rate=0.0
    lateinit var f:(Double)->Double
    constructor(rate:Double,f:(Double)->Double):this(){
        this.f=f
        this.rate=rate
    }
    fun computeGradient(cur1_x:Double,precision:Double,max_iters:Int,previous_step_size1:Double=1.0):Double{
        var iters:Int=0
        var cur_x=cur1_x
        var previous_step_size=previous_step_size1
        while (previous_step_size>precision && iters<max_iters){
            var prev_x=cur_x
            cur_x=cur_x-rate*this.f(prev_x)
            previous_step_size= abs(cur_x-prev_x)
            iters+=1
            println("Iteration $iters")
            println("x value is $cur_x")
        }
        println("The local minimum occurs at $cur_x")
        return cur_x
    }
}
fun xcontoh(x:Double):Double{
    return 2*(x+5)
}
fun main(){
    var obj3=GradientDescent(0.01,::xcontoh)
    obj3.computeGradient(3.0,0.0000001,10000)
}