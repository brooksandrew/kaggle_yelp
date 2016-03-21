package modeling.training

import modeling.processing.makeDataSets._
import modeling.io._

import java.util.Random
import java.nio.file._
import java.io.{DataOutputStream, File}
import org.apache.commons.io.FileUtils
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._

object cnn {
  
  def trainModel(alignedData: List[(Int, String, Vector[Int], Set[Int])], bizClass: Int = 1, saveNN: String = "") = {
    
    val ds = makeDataSet(alignedData, bizClass)

    println("commence training")
    println("class for training: " + bizClass)

    
    val begintime = System.currentTimeMillis()
  
    lazy val log = LoggerFactory.getLogger(cnn.getClass)
    log.info("Begin time: " + java.util.Calendar.getInstance().getTime())
  
      val nfeatures = ds.getFeatures.getRow(0).length // hyper, hyper parameter
      
      val numRows =  Math.sqrt(nfeatures).toInt // numRows * numColumns must equal columns in initial data * channels
      val numColumns = Math.sqrt(nfeatures).toInt // numRows * numColumns must equal columns in initial data * channels
      val nChannels = 1 // would be 3 if color image w R,G,B
      val outputNum = 2 // # of classes (# of columns in output)
      val iterations = 3
      val splitTrainNum = math.ceil(ds.numExamples*0.8).toInt // 80/20 training/test split
      val seed = 123
      val listenerFreq = 1
      val nOutPar = 50 // default was 1000.  # of output nodes in first layer
  
      println("rows: " + ds.getFeatures.size(0))
      println("columns: " + ds.getFeatures.size(1))
      
      /**
       *Set a neural network configuration with multiple layers
       */
      log.info("Load data....")
      ds.normalizeZeroMeanZeroUnitVariance() // this changes ds
      System.out.println("Loaded " + ds.labelCounts)
      Nd4j.shuffle(ds.getFeatureMatrix, new Random(seed), 1) // this changes ds.  Shuffles rows
      Nd4j.shuffle(ds.getLabels, new Random(seed), 1) // this changes ds.  Shuffles labels accordingly
      val trainTest: SplitTestAndTrain = ds.splitTestAndTrain(splitTrainNum, new Random(seed)) // Random Seed not needed here
  
      val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
              .seed(seed)
              .iterations(iterations)
              .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
              .list(2)
              .layer(0, new ConvolutionLayer.Builder(Array(1, 1):_*)
                      .nIn(nChannels)
                      .stride(2,2) // default stride(2,2)
                      .nOut(nOutPar)
                      .activation("relu")
                      .weightInit(WeightInit.RELU)
                      .build())
              .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                      .nOut(outputNum)
                      .weightInit(WeightInit.XAVIER)
                      .activation("softmax")
                      .build())
              .backprop(true).pretrain(false)
              
      new ConvolutionLayerSetup(builder, numRows, numColumns, nChannels)
              
      val conf: MultiLayerConfiguration = builder.build()

      log.info("Build model....")
      val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
      model.init() 
      model.setListeners(Seq[IterationListener](new ScoreIterationListener(listenerFreq)).asJava)

      log.info("Train model....")
      System.out.println("Training on " + trainTest.getTrain.labelCounts())
      model.fit(trainTest.getTrain)

      log.info("Evaluate model....")
      System.out.println("Testing on " + trainTest.getTest.labelCounts)
      val eval = new Evaluation(outputNum)
      val output: INDArray = model.output(trainTest.getTest.getFeatureMatrix) // these are the predictions
      eval.eval(trainTest.getTest.getLabels, output) // this changes the eval object (not to be confused w method w same name)
      log.info(eval.stats())
      
      val endtime = System.currentTimeMillis()
      log.info("End time: " + java.util.Calendar.getInstance().getTime())
      log.info("computation time: " + (endtime-begintime)/1000.0 + " seconds")
      
      log.info("Write results....")
      
      if(!saveNN.isEmpty) { 
        // model config
        FileUtils.write(new File(saveNN + ".json"), model.getLayerWiseConfigurations().toJson()) 
        
        // model parameters
        val dos: DataOutputStream = new DataOutputStream(Files.newOutputStream(Paths.get(saveNN + ".bin")))
        Nd4j.write(model.params(), dos)
      }
    
      log.info("****************Example finished********************")
          
  }

  
}