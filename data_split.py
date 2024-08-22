EXPT_NAME = 'expt_1'
TRAIN_FRACTION = 0.9

import datasets
import os
from tqdm import tqdm
import random


def random_stream_interleaver(stream_list):
    lens = [len(stream) for stream in stream_list]
    iters = [iter(stream) for stream in stream_list]
    num_items = sum(lens)
    while num_items > 0:
        story_idx = random.randint(0, num_items - 1)
        cum_sum = 0
        for i in range(len(lens)):
            cum_sum += lens[i]
            if cum_sum > story_idx:
                break
        yield next(iters[i])
        num_items -= 1
        lens[i] -= 1

def sanity_check_stream_interleave():
    import collections
    counts = collections.Counter()
    for i in range(10000):
        counts[tuple(random_stream_interleaver([[0], [1], [2,2]]))] += 1
    print(counts)
    print(''.join(list(random_stream_interleaver(['aaaa', 'b', 'ccccccccccc']))))
# sanity_check_stream_interleave()
def write_english_story_tinyprompt_str(text):
    return f'[PROMPT] english [USER] {text} [END]'
def write_english_story_tinyprompt(story_dict):
    return write_english_story_tinyprompt_str(story_dict['text'].strip())

def write_spanish_story_tinyprompt_str(text):
    return f'[PROMPT] español [USER] {text} [END]'
def write_spanish_story_tinyprompt(story_dict):
    return write_spanish_story_tinyprompt_str(story_dict["story"].strip())

def paragraph_splitter(text):
    return [t.strip() for t in text.split('\n') if t.strip()]

def write_translation_story_tinyprompt_strs(spanish_text, english_text):
    spanish_paragraphs = paragraph_splitter(spanish_text)
    translation_paragraphs = paragraph_splitter(english_text)

    if len(spanish_paragraphs) != len(translation_paragraphs):
        # print("count of spanish paragraphs did not match count of translation paragraphs")
        return ''
    
    alternating_paragraphs = ''.join(
        a + '\n' + b + '\n' for a, b in zip(translation_paragraphs, spanish_paragraphs))
    return f'[PROMPT] translate [USER] {alternating_paragraphs.strip()} [END]'

def write_translation_story_tinyprompt(story_dict):
    spanish_story = story_dict['story']
    translation = story_dict['translation']
    return write_translation_story_tinyprompt_strs(spanish_story, translation)
    

# print (paragraph_splitter('hello\n  \nworln\ntwo\nthree'))
# print(write_translation_story_tinyprompt(tinystories_ds_es_translated['train'][0]))
def function_applying_iterator(items, func):
    for item in items:
        yield func(item)

class FunctionApplyingIterable:
    def __init__(self, items, func):
        self.items = items
        self.func = func

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return function_applying_iterator(self.items, self.func)
    
if __name__ == '__main__':
    tinystories_ds_es_translated = datasets.load_dataset("robrenaud/multilingual_tinystories", data_files=["stories_00.json"])
    tinystories_ds_en = datasets.load_dataset("roneneldan/TinyStories")
    raw_es_stories = [f'stories_{i:02d}.json' for i in range(1, 22)]
    tinystories_ds_es = datasets.load_dataset("robrenaud/multilingual_tinystories", data_files=raw_es_stories)

    print('num translations   ', len(tinystories_ds_es_translated['train']))
    print('num spanish stories', len(tinystories_ds_es['train']))
    print('num english stories', len(tinystories_ds_en['train']))
    datasets_with_formatters = [
        ('english', tinystories_ds_en['train'].to_list(), write_english_story_tinyprompt),
        ('spanish', tinystories_ds_es['train'].to_list(), write_spanish_story_tinyprompt),
        ('translation', tinystories_ds_es_translated['train'].to_list(), write_translation_story_tinyprompt)
    ]

    interleavable_train_streams = []
    for task, ds, formatter in datasets_with_formatters:
        split_index = int(TRAIN_FRACTION * len(ds))
        interleavable_train_streams.append(FunctionApplyingIterable(ds[:split_index], formatter))

    os.makedirs(EXPT_NAME, exist_ok=True)
    with open(f'{EXPT_NAME}/train.txt', 'w') as f:
        for story in tqdm(random_stream_interleaver(interleavable_train_streams), desc='writing interleaved stories'):
            f.write(story + '\n')


    for task, ds, formatter in datasets_with_formatters:
        with open(f'{EXPT_NAME}/test_{task}.txt', 'w') as f:
            split_index = int(TRAIN_FRACTION * len(ds))
            for story in tqdm(ds[split_index:len(ds)], desc=f'writing {task} stories'):
                f.write(formatter(story) + '\n')


