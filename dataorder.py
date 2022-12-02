import numpy as np
import torch

from sklearn.utils import shuffle

def reorder_equal(train_loader):
    all_inputs, all_labels = [], []
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.cpu().detach().numpy(), labels.cpu().detach().numpy()
        batch_size = len(inputs)
        all_inputs.extend(inputs)
        all_labels.extend(labels)

    all_inputs, all_labels = np.array(all_inputs), np.array(all_labels)
    num_classes = len(np.unique(all_labels))
    # batch_per_class = int(batch_size/num_classes)
    # batch_size = batch_per_class * num_classes

    input_dict = {}
    for ele in range(num_classes):
        input_dict[ele] = all_inputs[all_labels==ele]

    new_inputs, new_labels = [], []
    iterator = 0
    data_left = True
    prev_len = 0
    while data_left:
        for ele in range(num_classes):
            if iterator < len(input_dict[ele]):
                new_inputs.append(input_dict[ele][iterator])
                new_labels.append(ele)

        if len(new_labels)==prev_len:
            break
        prev_len = len(new_labels)
        iterator += 1

    # new_inputs, new_labels = [], []
    # iterator = 0
    # data_left = True
    # prev_len = 0
    # while data_left:
    #     for ele in range(num_classes):
    #         st_ind, end_ind = iterator*batch_per_class, (iterator+1)*batch_per_class
    #         if len(input_dict[ele][st_ind:end_ind]) == 0:
    #             # data_left = False
    #             continue
    #         # if end_ind > len(input_dict[ele]):
    #         #     data_left = False
    #         new_inputs.extend(input_dict[ele][st_ind:end_ind])
    #         new_labels.extend([ele] * len(input_dict[ele][st_ind:end_ind]))
    #
    #     if len(new_labels)==prev_len:
    #         break
    #     prev_len = len(new_labels)
    #     iterator += 1

    print(iterator)
    # left_inputs, left_labels = [], []
    # for ele in range(num_classes):
    #     st_ind = iterator*batch_per_class
    #     if st_ind < len(input_dict[ele]):
    #         left_inputs.extend(input_dict[ele][st_ind:])
    #         left_labels.extend([ele] * len(input_dict[ele][st_ind:]))
    #
    # print(len(left_inputs))
    # left_inputs, left_labels = shuffle(left_inputs, left_labels, random_state=0)
    #
    # new_inputs.extend(left_inputs)
    # new_labels.extend(left_labels)

    new_inputs, new_labels = np.array(new_inputs[::-1]), np.array(new_labels[::-1])
    tensorx_train = torch.from_numpy(new_inputs)
    tensory_train = torch.from_numpy(new_labels)
    train_dataset = torch.utils.data.TensorDataset(tensorx_train, tensory_train)

    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
