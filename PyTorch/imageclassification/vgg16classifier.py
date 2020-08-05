import os
import time

import torch
import torchvision

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_FILE = 'vgg16_checkpoint_{}.pth'
RESULT_DIR = os.path.join(BASE_DIR, 'result')
RESULT_FILE = os.path.join(RESULT_DIR, 'vgg16_result.txt')


def init(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    vgg16 = torchvision.models.vgg16(pretrained=False, progress=True)
    vgg16.classifier[6].out_features = 10

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    )
    print('Loading training dataset ...')
    CIFAR_train_set = torchvision.datasets.CIFAR10(DATA_DIR, train=True, transform=transform, download=True)
    CIFAR_train_loader = torch.utils.data.DataLoader(CIFAR_train_set, batch_size=config['batch_size'], shuffle=True,
                                                     num_workers=config['num_worker'])


    print(f'training dataset size: {len(CIFAR_train_set)}')
    print(f'training sample size: {CIFAR_train_set[0][0].size()}')

    print('Loading testing dataset ...')
    CIFAR_test_set = torchvision.datasets.CIFAR10(DATA_DIR, train=False, transform=transform, download=True)
    CIFAR_test_loader = torch.utils.data.DataLoader(CIFAR_test_set, batch_size=config['batch_size']*2, shuffle=False,
                                                    num_workers=config['num_worker'])

    # CIFAR_train_loader = [(data[0].to(device), data[1].to(device)) for _, data in enumerate(CIFAR_train_loader)]
    # CIFAR_test_loader = [(data[0].to(device), data[1].to(device)) for _, data in enumerate(CIFAR_test_loader)]

    print(f'testing dataset size: {len(CIFAR_test_set)}')
    print(f'testing sample size: {CIFAR_test_set[0][0].size()}')

    return device, vgg16, CIFAR_train_loader, CIFAR_test_loader


def train(device, network, dataloader, config):
    print('\n training ...')

    network.to(device)
    network.train()

    optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    step = 1
    average_loss = 0
    average_correct = 0
    start_time = time.time()
    for e in range(config['epoch']):
        for i, data in enumerate(dataloader, 1):
            inputs, labels = data[0].to(device), data[1].to(device)
            # inputs, labels = data[0], data[1]

            optimizer.zero_grad()

            logits = network(inputs)

            _, predicts = torch.max(logits, 1)
            average_correct += (predicts == labels).sum().item() * 1.0 / labels.shape[0]

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            average_loss += loss.item()

            if step % config['print_every'] == 0:
                average_loss = average_loss / config['print_every']
                average_correct = average_correct / config['print_every']
                print('[epoch: %3d, step: %5d] loss: %.3f, accuracy: %3f' % (e + 1, i, average_loss, average_correct))
                average_loss = 0
                average_correct = 0

            if step % config['save_every'] == 0:
                model_file = os.path.join(MODEL_DIR, MODEL_FILE.format(step))
                torch.save(network.state_dict(), model_file)

            step += 1

    total_time = time.time() - start_time
    with open(RESULT_FILE, 'a') as f:
        f.write(f'\n\ntraining config: \n {config}')
        f.write(f'\ntraining results: \n')
        f.write('\n total training time: \t {}'.format(total_time))
        f.write('\n final average training lostt: \t {:.3f}'.format(average_loss))
        f.write('\n final average training accuracy: \t{:.3f}'.format(average_correct))


def test(device, network, dataloader, config):
    network.to(device)
    network.eval()

    correct = 0
    total = 0

    start_time = time.time()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            logits = network(inputs)
            _, predicts = torch.max(logits, 1)
            correct += (predicts == labels).sum().item() * 1.0
            total += len(inputs)
    total_time = time.time() - start_time

    s = '\ntesting results: \n accuracy on {} test images: {:.3f}, total time: {} s\n'.format(total,
                                                                                             correct * 1.0 / total,
                                                                                             total_time)
    print(s)
    with open(RESULT_FILE, 'a') as f:
        f.write(s)
