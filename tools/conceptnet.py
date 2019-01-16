import os
import json


def convert_to_eng():
    with open(CNET_ORIG, 'r', encoding='utf-8') as f:
        ls = f.readlines()
    en_ls = []
    for l in ls:
        fields = l.split()
        src = fields[2]
        dst = fields[3]
        if src.split('/')[2] == dst.split('/')[2] == 'en':
            en_ls.append(l)

    print(en_ls[0])

    with open(CNET_ENG, 'w', encoding='utf-8') as f:
        f.writelines(en_ls)


def main():
    with open(CNET_ENG, 'r', encoding='utf-8') as f:
        ls = f.readlines()
    red_ls = []
    for l in ls:
        fields = l.split()
        src = fields[2]
        dst = fields[3]
        assert src[:6] == dst[:6] == '/c/en/'
        src = src[6:]
        dst = dst[6:]
        other = ' '.join(fields[4:])
        other_dict = json.loads(other)
        weight = other_dict['weight']
        red_ls.append((src, dst, weight))
    print('\n'.join([str(l) for l in red_ls[:5]]))


if __name__ == '__main__':
    CNET_ENG = os.path.join('KB', 'Conceptnet', 'conceptnet560_en.txt')
    CNET_ORIG = os.path.join('KB', 'Conceptnet', 'conceptnet560.csv')
    # convert_to_eng()
    main()
