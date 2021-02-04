import * as tf from '@tensorflow/tfjs-node';
import { Logger } from 'sitka';
import * as fs from 'fs';
import { DoddleData } from './doodle-data.model';

export class Classifier {
  model: tf.Sequential;
  private data: DoddleData;
  private logger: Logger;

  constructor(data: DoddleData) {
    this.logger = Logger.getLogger({ name: this.constructor.name });

    this.data = data;
    this.model = tf.sequential();
    this.model.add(
      tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 3,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
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
        kernelSize: 3,
        filters: 32,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
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
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
      })
    );

    const optimizer = tf.train.adam();
    this.model.compile({
      optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
  }

  async train() {
    const training = tf.data
      .generator(() => this.data.dataGenerator(this.data, 'train'))
      .shuffle(this.data.maxImageClass * this.data.totalClasses)
      .batch(200);

    const test = tf.data
      .generator(() => this.data.dataGenerator(this.data, 'test'))
      .shuffle(this.data.maxImageClass * this.data.totalClasses)
      .batch(200);

    await this.model.fitDataset(training, {
      epochs: 2,
      validationData: test,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          this.logger.debug(
            `Epoch: ${epoch} - acc: ${logs?.acc.toFixed(
              3
            )} - loss: ${logs?.loss.toFixed(3)}`
          );
        }
      }
    });
  }

  async save() {
    fs.writeFileSync(
      'doddle-model-ts/classes.json',
      JSON.stringify({ classes: this.data.classes })
    );
    await this.model.save('file://./doddle-model-ts');
  }
}
