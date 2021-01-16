import * as tf from '@tensorflow/tfjs';
import { Logger } from 'sitka';
import * as fs from 'fs';

export class DoddleData {
  readonly IMAGE_SIZE = 784;
  readonly TRAIN_TEST_RATIO = 8 / 10;
  datasetImages = new Float32Array();
  datasetLabels = new Uint8Array();
  trainImages = new Float32Array();
  trainLabels = new Uint8Array();
  testImages = new Float32Array();
  testLabels = new Uint8Array();
  trainIndices = new Uint32Array();
  testIndices = new Uint32Array();
  numTrainElements = 0;
  numTestElements = 0;
  totalImages = 0;
  totalLabels = 0;
  labels: string[] = [];
  private shuffledTrainIndex = 0;
  private shuffledTestIndex = 0;
  private logger: Logger;

  constructor() {
    this.logger = Logger.getLogger({ name: this.constructor.name });
  }

  async loadData(paths: string[], labels: string[]) {
    if (paths.length !== labels.length) {
      throw new Error('paths and labels must have the same lenght');
    }
    this.labels = labels;
    this.totalLabels = labels.length;
    const data: {
      [label: string]: { data: Float32Array; totalImages: number };
    } = {};

    // tslint:disable-next-line: prefer-for-of
    for (let i = 0; i < paths.length; i++) {
      const path = paths[i];
      const label = labels[i];

      // NOTE: 80 is length of the header

      const bytes = new Uint8Array(fs.readFileSync(path, null).buffer).slice(
        80
      );
      const bytesToFloat = new Float32Array(bytes).map((x) => x / 255);

      data[label] = {
        data: bytesToFloat,
        totalImages: bytes.length / this.IMAGE_SIZE
      };
      // const first = data[label].data.slice(0, 784);
      // fs.writeFile(`src/data/${label}-first.txt`, first.toString(), () => null);
      this.totalImages += data[label].totalImages;
      this.logger.debug(`${label} loaded!`);
    }

    this.numTrainElements = Math.floor(
      this.TRAIN_TEST_RATIO * this.totalImages
    );
    this.numTestElements = this.totalImages - this.numTrainElements;

    this.datasetImages = new Float32Array(this.totalImages * this.IMAGE_SIZE);
    this.datasetLabels = new Uint8Array(this.totalImages * this.totalLabels);

    let offset = 0;
    // tslint:disable-next-line: forin
    for (const label in data) {
      this.datasetImages.set(data[label].data, offset * this.IMAGE_SIZE);
      const itemLabel = new Uint8Array(this.totalLabels).fill(0);
      itemLabel[this.labels.indexOf(label)] = 1;
      for (let i = 0; i < data[label].totalImages; i++) {
        this.datasetLabels.set(itemLabel, (i + offset) * this.totalLabels);
      }
      offset += data[label].totalImages;
    }

    this.trainIndices = tf.util.createShuffledIndices(this.numTrainElements);
    this.testIndices = tf.util.createShuffledIndices(this.numTestElements);

    this.trainImages = this.datasetImages.slice(
      0,
      this.IMAGE_SIZE * this.numTrainElements
    );
    this.testImages = this.datasetImages.slice(
      this.IMAGE_SIZE * this.numTrainElements
    );
    this.trainLabels = this.datasetLabels.slice(
      0,
      this.totalLabels * this.numTrainElements
    );
    this.testLabels = this.datasetLabels.slice(
      this.totalImages * this.numTrainElements
    );

    this.logger.debug('All data loaded');
  }

  nextTrainBatch(batchSize: number) {
    return this.nextBatch(
      batchSize,
      [this.trainImages, this.trainLabels],
      () => {
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length;
        return this.trainIndices[this.shuffledTrainIndex];
      }
    );
  }

  nextTestBatch(batchSize: number) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
        (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(
    batchSize: number,
    data: (Uint8Array | Float32Array)[],
    indexFn: () => number
  ) {
    const batchImagesArray = new Float32Array(batchSize * this.IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * this.totalLabels);

    for (let i = 0; i < batchSize; i++) {
      const index = indexFn();
      const image = data[0].slice(
        index * this.IMAGE_SIZE,
        index * this.IMAGE_SIZE + this.IMAGE_SIZE
      );
      batchImagesArray.set(image, i * this.IMAGE_SIZE);

      const label = data[1].slice(
        index * this.totalLabels,
        index * this.totalLabels + this.totalLabels
      );
      batchLabelsArray.set(label, i * this.totalLabels);
    }

    const xs = tf.tensor2d(batchImagesArray, [batchSize, this.IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, this.totalLabels]);

    return { xs, labels };
  }

  // shuffle() {
  //   const order = tf.util.createShuffledIndices(this.totalImages);

  //   const shuffledData = new Uint8Array(this.datasetImages.length);
  //   const shuffledLabels = new Uint8Array(this.datasetLabels.length);

  //   for (let i = 0; i < this.totalImages; i++) {
  //     const index = order[i];
  //     const data = this.datasetImages.slice(
  //       index * this.IMAGE_SIZE,
  //       (index + 1) * this.IMAGE_SIZE
  //     );
  //     const label = this.datasetLabels.slice(
  //       index * this.totalLabels,
  //       (index + 1) * this.totalLabels
  //     );
  //     shuffledData.set(data, i * this.IMAGE_SIZE);
  //     shuffledLabels.set(label, i * this.totalLabels);
  //   }

  //   this.datasetImages = shuffledData;
  //   this.datasetLabels = shuffledLabels;
  // }

  // getTrainBatch(batchSize: number, offset: number) {
  //   const batchData = this.datasetImages.slice(
  //     offset * this.IMAGE_SIZE,
  //     (batchSize + offset) * this.IMAGE_SIZE
  //   );
  //   const batchLabels = this.datasetLabels.slice(
  //     offset * this.totalLabels,
  //     (batchSize + offset) * this.totalLabels
  //   );
  //   const batch = {
  //     data: tf.tensor2d(Array.from(batchData), [
  //       batchData.length / this.IMAGE_SIZE,
  //       this.IMAGE_SIZE
  //     ]),
  //     labels: tf.tensor2d(Array.from(batchLabels), [
  //       batchLabels.length / this.totalLabels,
  //       this.totalLabels
  //     ]),
  //     size: batchData.length / this.IMAGE_SIZE
  //   };
  //   return batch;
  // }

  // getFirst() {
  //   const first = tf.tensor2d(this.datasetImages.slice(0, this.IMAGE_SIZE), [
  //     1,
  //     this.IMAGE_SIZE
  //   ]);

  //   return first.reshape([1, 28, 28, 1]);
  // }
}
