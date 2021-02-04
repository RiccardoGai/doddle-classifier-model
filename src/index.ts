import { Logger } from 'sitka';
import { Classifier } from './model/classifier.model';
import { DoddleData } from './model/doodle-data.model';

const logger = Logger.getLogger({ name: 'index' });

async function main() {
  const data = new DoddleData({
    directoryData: 'src/data',
    maxImageClass: 20000
  });
  data.loadData();

  const model = new Classifier(data);
  await model.train();
  logger.debug('model is ready!');

  await model.save();
  logger.debug('model is saved!');
}

main();
