import { Classifier } from './model/classifier.model';
import { DoddleData } from './model/doodle-data.model';
import { Logger } from 'sitka';
const logger = Logger.getLogger({ name: 'index' });

async function main() {
  const data = new DoddleData();
  await data.loadData(
    ['src/data/cat.npy', 'src/data/airplane.npy', 'src/data/train.npy'],
    ['cat', 'airplane', 'train']
  );
  data.shuffle();

  const model = new Classifier(data);
  await model.train();
  logger.debug('model is ready!');

  await model.save();
}

main();
