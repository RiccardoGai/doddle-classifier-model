import * as tf from '@tensorflow/tfjs';
import { Logger } from 'sitka';
import * as fs from 'fs';

const IMAGE_SIZE = 784;

export class DoddleData {
  trainingData: Uint8Array = new Uint8Array();
  trainingLabels: Uint8Array = new Uint8Array();
  totalImages: number = 0;
  totalClasses: number = 0;
  labels: string[] = [];
  private _logger: Logger;

  constructor() {
    this._logger = Logger.getLogger({ name: this.constructor.name });
  }

  async loadData(paths: string[], labels: string[]) {
    if (paths.length !== labels.length) {
      throw new Error('paths and labels must have the same lenght');
    }
    this.labels = labels;
    this.totalClasses = labels.length;
    const data: {
      [label: string]: { data: Uint8Array; totalImages: number };
    } = {};

    // tslint:disable-next-line: prefer-for-of
    for (let i = 0; i < paths.length; i++) {
      const path = paths[i];
      const label = labels[i];
      const t = new Uint8Array(fs.readFileSync(path, null).buffer);
      const bytes = t.slice(80, t.length);
      data[label] = {
        data: bytes,
        totalImages: bytes.length / IMAGE_SIZE
      };
      this.totalImages += data[label].totalImages;
      this._logger.debug(`${label} loaded!`);
    }

    this.trainingData = new Uint8Array(this.totalImages * IMAGE_SIZE);
    this.trainingLabels = new Uint8Array(this.totalImages * this.totalClasses);

    let offset = 0;
    // tslint:disable-next-line: forin
    for (const label in data) {
      this.trainingData.set(data[label].data, offset * IMAGE_SIZE);
      const itemLabel = new Uint8Array(this.totalClasses).fill(0);
      itemLabel[this.labels.indexOf(label)] = 1;
      for (let i = 0; i < data[label].totalImages; i++) {
        this.trainingLabels.set(itemLabel, (i + offset) * this.totalClasses);
      }
      offset += data[label].totalImages;
    }
    this._logger.debug('All data loaded');
  }

  shuffle() {
    const order = tf.util.createShuffledIndices(this.totalImages);

    const shuffledData = new Uint8Array(this.trainingData.length);
    const shuffledLabels = new Uint8Array(this.trainingLabels.length);

    for (let i = 0; i < this.totalImages; i++) {
      const index = order[i];
      const data = this.trainingData.slice(
        index * IMAGE_SIZE,
        (index + 1) * IMAGE_SIZE
      );
      const label = this.trainingLabels.slice(
        index * this.totalClasses,
        (index + 1) * this.totalClasses
      );
      shuffledData.set(data, i * IMAGE_SIZE);
      shuffledLabels.set(label, i * this.totalClasses);
    }

    this.trainingData = shuffledData;
    this.trainingLabels = shuffledLabels;
  }

  getTrainBatch(batchSize: number, offset: number) {
    const batchData = this.trainingData.slice(
      offset * IMAGE_SIZE,
      (batchSize + offset) * IMAGE_SIZE
    );
    const batchLabels = this.trainingLabels.slice(
      offset * this.totalClasses,
      (batchSize + offset) * this.totalClasses
    );
    const batch = {
      data: tf.tensor2d(Array.from(batchData), [
        batchData.length / IMAGE_SIZE,
        IMAGE_SIZE
      ]),
      labels: tf.tensor2d(Array.from(batchLabels), [
        batchData.length / IMAGE_SIZE,
        this.totalClasses
      ]),
      size: batchData.length / IMAGE_SIZE
    };
    return batch;
  }

  getFirst() {
    const first = tf.tensor2d(this.trainingData.slice(0, IMAGE_SIZE), [
      1,
      IMAGE_SIZE
    ]);

    return first.reshape([1, 28, 28, 1]);
  }
}
