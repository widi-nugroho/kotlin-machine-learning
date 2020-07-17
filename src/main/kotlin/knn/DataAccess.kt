package knn
import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import java.io.File
import java.io.BufferedReader
object DataAccess {
    fun loadcsv(filename:String):List<Data>{
        var res= mutableListOf<Data>()
        csvReader().open(filename) {
            readAllAsSequence().forEach { row ->
                var output=row[row.lastIndex]
                var input= listOf(row[0].toFloat(),row[1].toFloat(),row[2].toFloat(),row[3].toFloat())
                var satudata=Data(input,output)
                res.add(satudata)
            }
        }
        println(res[0].input)
        return res
    }
}
fun main (){
    DataAccess.loadcsv("/home/widi/projects/kotlin-machine-learning/src/main/resources/iris.data")
}