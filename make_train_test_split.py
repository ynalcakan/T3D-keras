from random import shuffle
import csv
import glob

action_classes = ['Cut-in', 'LanePass']


def create_csvs():
    train = []
    test = []

    for myclass, directory in enumerate(action_classes):
        for filename in glob.glob('data/train/{}/*.avi'.format(directory)):
            train.append([filename, myclass, directory])

    for myclass, directory in enumerate(action_classes):
        for filename in glob.glob('data/test/{}/*.avi'.format(directory)):
            test.append([filename, myclass, directory])

    shuffle(train)
    shuffle(test)
    # print('train', len(total_train))
    # print('test', len(total_test))

    with open('train.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(['path', 'class', 'action'])
        mywriter.writerows(train)
        print('Training CSV file created successfully')

    with open('test.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(['path', 'class', 'action'])
        mywriter.writerows(test)
        print('Testing CSV file created successfully')

    print('CSV files created successfully')


if __name__ == "__main__":
    create_csvs()
