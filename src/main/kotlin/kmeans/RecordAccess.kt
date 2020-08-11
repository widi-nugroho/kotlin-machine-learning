package kmeans

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader

object RecordAccess {
    fun loadcsv(filename:String):List<Record>{
        var res= mutableListOf<Record>()

        csvReader().open(filename) {
            readAllAsSequence().forEach { row ->
                var output=row[row.lastIndex]
                var resmap= mutableMapOf<String,Double>()
                resmap["sepal_L"]=row[0].toDouble()
                resmap["sepal_W"]=row[1].toDouble()
                resmap["petal_L"]=row[2].toDouble()
                resmap["petal_W"]=row[3].toDouble()

                var satudata= Record(resmap,output)
                res.add(satudata)
            }
        }
        return res
    }
}
