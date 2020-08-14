package learn_tsne

import com.jujutsu.tsne.barneshut.BHTSne
import com.jujutsu.tsne.barneshut.BarnesHutTSne
import com.jujutsu.tsne.barneshut.ParallelBHTsne
import com.jujutsu.utils.MatrixOps
import com.jujutsu.utils.MatrixUtils
import com.jujutsu.utils.TSneUtils
import java.io.File
import org.math.plot.Plot2DPanel
import javax.swing.JFrame


object TSneTest {
    @JvmStatic
    fun main(args: Array<String>) {
        val initial_dims = 55
        val perplexity = 20.0
        val X =
            MatrixUtils.simpleRead2DMatrix(File("/home/widi/projects/kotlin-machine-learning/src/main/resources/mnist2500_X.txt"), "   ")
        println(X)
        println(MatrixOps.doubleArrayToPrintString(X, ", ", 50, 10))
        val tsne: BarnesHutTSne
        val parallel = false
        tsne = if (parallel) {
            ParallelBHTsne()
        } else {
            BHTSne()
        }
        val config = TSneUtils.buildConfig(X, 2, initial_dims, perplexity, 10000)
        val Y = tsne.tsne(config)
        var plot=Plot2DPanel()
        val frame = JFrame("tSNE Plot")
        frame.setSize(600, 600)
        var x= mutableListOf<Double>()
        var y= mutableListOf<Double>()
        for (i in Y){
            x.add(i[0])
            y.add(i[1])
        }
        plot.addScatterPlot("mnist",x.toDoubleArray(),y.toDoubleArray())
        frame.contentPane=plot
        frame.isVisible=true
        // Plot Y or save Y to file and plot with some other tool such as for instance R

    }
}