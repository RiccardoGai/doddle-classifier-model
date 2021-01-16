import { Classifier } from './model/classifier.model';
import { DoddleData } from './model/doodle-data.model';
import { Logger } from 'sitka';
import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
const logger = Logger.getLogger({ name: 'index' });

async function main() {
  const data = new DoddleData();
  await data.loadData(
    ['src/data/cat.npy', 'src/data/airplane.npy', 'src/data/train.npy'],
    ['cat', 'airplane', 'train']
  );
  // data.shuffle();

  const model = new Classifier(data);
  await model.train();
  logger.debug('model is ready!');

  await model.save();

  const catFirst = new Float32Array(
    fs
      .readFileSync('src/data/cat-first.txt')
      .toString()
      .split(',')
      .map((x) => +x)
  );

  logger.debug(await model.predict(tf.tensor2d(catFirst, [1, 784])));
  const airplaneFirst = new Float32Array(
    fs
      .readFileSync('src/data/airplane-first.txt')
      .toString()
      .split(',')
      .map((x) => +x)
  );
  logger.debug(await model.predict(tf.tensor2d(airplaneFirst, [1, 784])));
  const trainFirst = new Float32Array(
    fs
      .readFileSync('src/data/train-first.txt')
      .toString()
      .split(',')
      .map((x) => +x)
  );
  logger.debug(await model.predict(tf.tensor2d(trainFirst, [1, 784])));
}

main();
