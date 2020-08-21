package ann

import java.awt.image.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;
import ai.djl.*;
import ai.djl.basicmodelzoo.basic.*;
import ai.djl.ndarray.*;
import ai.djl.modality.*;
import ai.djl.modality.cv.*;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.translate.*;


fun main(){
    var img = ImageFactory.getInstance().fromUrl("https://djl-ai.s3.amazonaws.com/resources/images/0.png")
    img.getWrappedImage()
    val modelDir: Path = Paths.get("build/mlp")
    val model = Model.newInstance("mlp")
    model.block = buatBlock(28*28,10)
    model.load(modelDir)
    val translator=MyTranslator()
    var predictor = model.newPredictor(translator);
    var classifications = predictor.predict(img)
    println(classifications)
}