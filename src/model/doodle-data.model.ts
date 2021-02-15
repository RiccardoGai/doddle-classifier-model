import * as tf from '@tensorflow/tfjs-node';
// import { Logger } from 'sitka';
import * as fs from 'fs';
import * as path from 'path';

export class DoddleData {
  readonly IMAGE_SIZE = 784;
  readonly IMAGE_WIDTH = 28;
  readonly IMAGE_HEIGHT = 28;
  readonly TRAIN_TEST_RATIO = 8 / 10;

  directoryData: string;
  maxImageClass: number;
  paths: string[] = [];
  classes: string[] = [];
  get totalClasses() {
    return this.classes.length;
  }

  constructor(obj: { directoryData: string; maxImageClass: number }) {
    this.directoryData = obj.directoryData;
    this.maxImageClass = obj.maxImageClass;
  }

  loadData() {
    this.paths = fs
      .readdirSync(this.directoryData)
      .filter((x) => x.endsWith('.npy'));

    if (!this.paths.length) {
      throw new Error('no .npy files found');
    }
    this.classes = this.paths.map((x) => x.replace('.npy', ''));
  }

  *dataGenerator(mode: 'train' | 'test') {
    // const logger = Logger.getLogger({ name: '*dataGenerator' });
    const offset = 255;
    // tslint:disable-next-line: prefer-for-of
    for (let i = 0; i < this.paths.length; i++) {
      let bytes = new Uint8Array(
        fs.readFileSync(
          path.join(this.directoryData, this.paths[i]),
          null
        ).buffer
      )
        .slice(80)
        .slice(0, this.maxImageClass * this.IMAGE_SIZE);

      const numImages = bytes.length / this.IMAGE_SIZE;

      if (mode === 'train') {
        bytes = bytes.slice(
          0,
          Math.floor(numImages * this.TRAIN_TEST_RATIO) * this.IMAGE_SIZE
        );
      } else if (mode === 'test') {
        bytes = bytes.slice(
          Math.floor(numImages * this.TRAIN_TEST_RATIO) * this.IMAGE_SIZE
        );
      }

      const label = this.classes[i];
      // logger.debug(`looping ${label}`);

      for (let j = 0; j < bytes.length; j = j + this.IMAGE_SIZE) {
        const singleImage = bytes.slice(j, j + this.IMAGE_SIZE);
        const image = tf
          .tensor(singleImage)
          .reshape([this.IMAGE_WIDTH, this.IMAGE_HEIGHT, 1])
          .toFloat();
        const xs = image.div(offset);
        const ys = tf.tensor(this.classes.map((x) => (x === label ? 1 : 0)));
        yield {
          xs,
          ys
        };
      }
    }
  }
}
