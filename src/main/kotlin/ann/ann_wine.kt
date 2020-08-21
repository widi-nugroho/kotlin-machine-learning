package ann

import ai.djl.Model
import ai.djl.basicdataset.Mnist
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Activation
import ai.djl.nn.Blocks
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.EasyTrain
import ai.djl.training.dataset.ArrayDataset
import ai.djl.training.evaluator.Accuracy
import ai.djl.training.listener.TrainingListener
import ai.djl.training.loss.Loss
import ai.djl.training.util.ProgressBar
import knn.DataAccess
import java.nio.file.Files
import java.nio.file.Paths

fun split(w:List<WineRecord>,m: NDManager):Pair<NDArray,NDArray>{
    var input= mutableListOf<Double>()
    var output= mutableListOf<Double>()
    for (i in w){
        for (p in i.input){
            input.add(p)
        }
        for (p in i.output){
            output.add(p)
        }
    }
    var datas=m.create(input.toDoubleArray()).reshape(w.size.toLong(),13)
    var labels=m.create(output.toDoubleArray()).reshape(w.size.toLong(),3)
    return Pair(datas,labels)
}
fun wineBlock(i:Long): SequentialBlock {
    val block = SequentialBlock()
    block.add(Blocks.batchFlattenBlock(i))
    block.add(Linear.builder().setOutChannels(9).build())
    block.add(Activation::relu)
    block.add(Linear.builder().setOutChannels(3).build())
    return block
}
fun main(){
    var manager=NDManager.newBaseManager()
    var wineDataset=DataAccess.WineList("/home/widi/projects/kotlin-machine-learning/src/main/resources/wine.data")
    var p=split(wineDataset,manager)
    var tes=NDList()
    var t=ArrayDataset::Builder
    var builder=t()
    builder.setData(p.first)
    builder.optLabels(p.second)
    builder.setSampling(2,false)
    var wine=builder.build()

    var batchsize=2
    var model = Model.newInstance("mlp")
    model.block= wineBlock(13)
    //model.block = Mlp(28 * 28, 10, intArrayOf(128, 64))
    val config =
            DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss()) //softmaxCrossEntropyLoss is a standard loss for classification problems
                    .addEvaluator(Accuracy()) // Use accuracy so we humans can understand how accurate the model is
                    .addTrainingListeners(*TrainingListener.Defaults.logging())

// Now that we have our training configuration, we should create a new trainer for our model

// Now that we have our training configuration, we should create a new trainer for our model
    val trainer = model.newTrainer(config)
    trainer.initialize(Shape(2,13))
    // Deep learning is typically trained in epochs where each epoch trains the model on each item in the dataset once.
    // Deep learning is typically trained in epochs where each epoch trains the model on each item in the dataset once.
    val epoch = 2

    for (i in 0 until epoch) {
        val index = 0

        // We iterate through the dataset once during this epoch
        for (batch in trainer.iterateDataset(wine)) {

            // During trainBatch, we update the loss and evaluators with the results for the training batch.
            EasyTrain.trainBatch(trainer, batch)

            // Now, we update the model parameters based on the results of the latest trainBatch
            trainer.step()

            // We must make sure to close the batch to ensure all the memory associated with the batch is cleared quickly.
            // If the memory isn't closed after each batch, you will very quickly run out of memory on your GPU
            batch.close()
        }
        // reset training and validation evaluators at end of epoch
        trainer.notifyListeners { listener: TrainingListener ->
            listener.onEpoch(
                    trainer
            )
        }
    }
    var modelDir = Paths.get("build/mlp")
    Files.createDirectories(modelDir)
    model.setProperty("Epoch",epoch.toString())
    model.save(modelDir, "mlp")
    model
}