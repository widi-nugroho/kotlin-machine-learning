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

class MyTranslator:Translator<Image,Classifications> {
    override fun getBatchifier(): Batchifier {
        return Batchifier.STACK
    }

    override fun processInput(ctx: TranslatorContext, input: ai.djl.modality.cv.Image): NDList {
        // Convert Image to NDArray
        val array = input.toNDArray(ctx.ndManager, ai.djl.modality.cv.Image.Flag.GRAYSCALE)
        return NDList(NDImageUtils.toTensor(array))
    }

    override fun processOutput(ctx: TranslatorContext, list: NDList): Classifications {
        val probabilities = list.singletonOrThrow().softmax(0)
        val indices = IntStream.range(0, 10).mapToObj { i: Int -> i.toString() }.collect(Collectors.toList())
        return Classifications(indices, probabilities)
    }

}