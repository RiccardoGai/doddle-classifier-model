import * as tf from '@tensorflow/tfjs-node';
import { Tensor } from '@tensorflow/tfjs-node';
import { Logger } from 'sitka';
import { DoddleData } from './doodle-data.model';

export class Classifier {
  model: tf.Sequential;
  private data: DoddleData;
  private _logger: Logger;

  constructor(data: DoddleData) {
    this._logger = Logger.getLogger({ name: this.constructor.name });

    this.data = data;
    this.model = tf.sequential();
    this.model.add(
      tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
      })
    );
    this.model.add(
      tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
      })
    );
    this.model.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'VarianceScaling'
      })
    );
    this.model.add(
      tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
      })
    );
    this.model.add(tf.layers.flatten());
    this.model.add(
      tf.layers.dense({
        units: this.data.totalClasses,
        kernelInitializer: 'VarianceScaling',
        activation: 'softmax'
      })
    );

    const LEARNING_RATE = 0.15;
    const optimizer = tf.train.sgd(LEARNING_RATE);
    this.model.compile({
      optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
  }

  async train() {
    const batchSize = 100;
    const iterations = this.data.totalImages / batchSize;
    for (let i = 0; i < iterations; i++) {
      const batch = this.data.getTrainBatch(batchSize, i * batchSize);

      // this._logger.debug(batch.data.dataSync());
      const batchData = batch.data.reshape([batch.size, 28, 28, 1]);
      // this._logger.debug(batchData.dataSync());
      const batchLabels = batch.labels;
      const options = {
        batchSize: batch.size,
        epochs: 1
      };

      const history = await this.model.fit(batchData, batchLabels, options);
      const loss = history.history.loss[0];
      const accuracy = history.history.acc[0];
      this._logger.debug(`batch: ${i} loss: ${loss} accuracy: ${accuracy}`);
    }
  }

  async predict(data: tf.Tensor) {
    const res = this.model.predict(data) as Tensor;
    return res.data();
  }

  async save() {
    await this.model.save('file://./doddle-model-ts');
  }
}
