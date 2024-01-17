import os
import sys
import json
import time
import spacy
import random
import numpy as np
import platform
import gpt_2_simple as gpt2
import tensorflow as tf

init_time = time.time()
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess, run_name="run_test")

st_time = time.time()


prefix = "It's OK to feel depressed,"
print("Inputting...")

answers = gpt2.generate(sess=sess,
                        run_name="run_test",
                        length=30,
                        temperature=0.7,
                        seed=random.randint(1, 1000),
                        prefix=prefix,
                        nsamples=5,
                        top_k=10,
                        top_p=0.9,
                        include_prefix=False,
                        truncate='.\n',
                        return_as_list=True,
                        batch_size=5)

print(answers[random.randint(0, 4)])
end_time = time.time()

print("Time Spent on load: %s" % (st_time-init_time))
print("Time Spent on generate: %s" % (end_time-st_time))
