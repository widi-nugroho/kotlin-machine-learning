package knn
import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
object DataAccess {
    fun loadcsv(filename:String):List<DataClassificator>{
        var res= mutableListOf<DataClassificator>()
        csvReader().open(filename) {
            readAllAsSequence().forEach { row ->
                var output=row[row.lastIndex]
                var input= listOf(row[0].toDouble(),row[1].toDouble(),row[2].toDouble(),row[3].toDouble())
                var satudata=DataClassificator(input,output)
                res.add(satudata)
            }
        }
        return res
    }
    fun loadAirFoilNoise(filename:String):List<DataRegressor>{
        var res= mutableListOf<DataRegressor>()
        csvReader().open(filename) {
            readAllAsSequence().forEach { row ->
                var output=row[row.lastIndex].toDouble()
                var input= listOf(row[0].toDouble(),row[1].toDouble(),row[2].toDouble(),row[3].toDouble(),row[4].toDouble())
                var satudata=DataRegressor(input,output)
                res.add(satudata)
            }
        }
        return res
    }
}
fun main (){
    DataAccess.loadcsv("/home/widi/projects/kotlin-machine-learning/src/main/resources/iris.data")
    DataAccess.loadAirFoilNoise("/home/widi/projects/kotlin-machine-learning/src/main/resources/airfoil_self_noise.csv")
}