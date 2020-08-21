package ann

import ai.djl.nn.Activation
import ai.djl.nn.Blocks
import ai.djl.nn.SequentialBlock
import ai.djl.nn.core.Linear
import ai.djl.*;
import ai.djl.nn.*;
import ai.djl.nn.core.*;
import ai.djl.training.*;


fun main(){
    val inputSize = 28 * 28.toLong()
    val outputSize: Long = 10
    val block = SequentialBlock()
    block.add(Blocks.batchFlattenBlock(inputSize));
    block.add(Linear.builder().setOutChannels(128).build());
    block.add(Activation::relu);
    block.add(Linear.builder().setOutChannels(64).build());
    block.add(Activation::relu);
    block.add(Linear.builder().setOutChannels(outputSize).build());
    block


}