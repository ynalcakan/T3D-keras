from random import shuffle
import csv
import glob

action_classes = ['Cut-in', 'LanePass']


def create_csvs():
    train = []
    test = []

    for myclass, directory in enumerate(action_classes):
        for filename in glob.glob('Data/{}/*.avi'.format(directory)):
            group = ((filename.split('/')[-1]).split('.')[0]).split('_')[-2]

            if group in ['g01', 'g02', 'g02', 'g04', 'g05']:
                test.append([filename, myclass, directory])
            else:
                train.append([filename, myclass, directory])

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
