
import json 
import regex as re
from tqdm import tqdm
from typing import Iterable, List, Optional
import base64

############################################################

def most_common_bp(vocabulary, byte_pair_frequencies):
    
    most_common_pair = max(byte_pair_frequencies.values())
    most_common_pairs_overall = []
    for pair, freq in byte_pair_frequencies.items():
        if freq == most_common_pair:
            most_common_pairs_overall.append(pair)

    ###################################################

    most_common_bp_found = []
    for pair in most_common_pairs_overall:
        most_common_bp_found.append((vocabulary[pair[0]], vocabulary[pair[1]]))

    ###################################################

    most_frequent_byte_pair = max(most_common_bp_found)
    for byte_pair, pair in zip(most_common_bp_found, most_common_pairs_overall):
        if most_frequent_byte_pair == byte_pair:
            return pair

############################################################

def update_frequencies(new_byte_pair, merged_id, token_freq, bp_freq, 
                       bp_tokens, token, id_before_merge, id_after_merge):

    for count in range(len(merged_id)):
        
        if merged_id[count] == new_byte_pair:
            
            if count <= 0:
                byte_before = None
            else:
                byte_before = merged_id[count - 1]

            if count + 1 >= len(merged_id):
                byte_after = None
            else:
                byte_after = merged_id[count + 1]
            
            #################################################

            if byte_before:

                prior_pair = (byte_before, id_before_merge)
                bp_freq[prior_pair] = bp_freq[prior_pair] - token_freq.get(token, 0)

                #################################################
                
                post_pair = (byte_before, new_byte_pair)
                bp_freq[post_pair] = bp_freq[post_pair] + token_freq.get(token, 0)

                #################################################
                
                if post_pair not in bp_tokens.keys():
                    bp_tokens[post_pair] = [token]
                else:
                    bp_tokens[post_pair].append(token) if token not in bp_tokens[post_pair] else None

            #################################################

            if byte_after:

                after_pair = (id_after_merge, byte_after)
                bp_freq[after_pair] = bp_freq[after_pair] - token_freq.get(token, 0)

                new_pair_after = (new_byte_pair, byte_after)
                bp_freq[new_pair_after] = token_freq.get(token, 0) + bp_freq[new_pair_after]

                #################################################

                if new_pair_after not in bp_tokens.keys():
                    bp_tokens[new_pair_after] = [token]
                else:
                    bp_tokens[new_pair_after].append(token) if token not in bp_tokens[new_pair_after] else None

                #################################################

    return bp_freq, bp_tokens

############################################################

def add_new_tokens(current_token, current_pair, bp_tokens):

    if current_pair not in bp_tokens.keys():
        bp_tokens[current_pair] = [current_token]
        return bp_tokens
    else:
        if current_token not in bp_tokens[current_pair]:
            bp_tokens[current_pair].append(current_token)
        return bp_tokens

############################################################

class Tokenizer():

    def __init__(self, vocabulary, merges, special_tokens=None):

        self.special_tokens = special_tokens if special_tokens is not None else []
        self.merges = {}
        for count, merge in enumerate(merges):
            self.merges.update({tuple(merge): count})

        #with open(bytes_file, 'r', encoding='utf-8') as f:
        #    loaded_encoded_tuples = json.load(f)
        #decoded_tuples = [(base64.b64decode(item[0]), base64.b64decode(item[1])) for item in loaded_encoded_tuples]
        
        self.vocabulary = vocabulary.copy()
        #with open(vocab_file, 'rb') as file:
        #    file_content = file.read()
        #bytes_list = file_content.strip().split(b'\n')

        self.inverse_vocab = {}
        for k, v in self.vocabulary.items():
            self.inverse_vocab[v] = k

    ############################################################
    
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None):
        
        vocabulary = {}
        with open(vocab_filepath, encoding='utf-8') as vocab_file:
            for row, line in enumerate(vocab_file):
                vocabulary[line.strip()] = row

        with open(merges_filepath, encoding='utf-8') as merges_file:
            for line in merges_file:
                merges = [tuple(line.strip().split())]

        return cls(vocabulary, merges, special_tokens)
    
    ############################################################
    
    def apply_merge_to_token(self, token_id_sequence, merge_pair, vocabulary) -> List[int]:
        
        count = 0
        while count < len(token_id_sequence) - 1:

            if (self.inverse_vocab[merge_pair[0]], self.inverse_vocab[merge_pair[1]]) == (token_id_sequence[count], token_id_sequence[count + 1]):
                token_id_sequence = token_id_sequence[:count] + [self.inverse_vocab[merge_pair[0] + merge_pair[1]]] + token_id_sequence[count + 2:]
            else:
                count += 1

        return token_id_sequence
    
    ############################################################
    
    def apply_BPE_merges(self, token: str, merges, vocab) -> List[int]:
        
        token_id_sequence = []

        if token not in self.special_tokens:
            for token_character in token:
                for token_byte in token_character.encode('utf-8'):
                    token_id_sequence.append(self.inverse_vocab[bytes([token_byte])])
        
        merged_id = token_id_sequence.copy()
        for merge_pair in merges:
            merged_id = self.apply_merge_to_token(merged_id, merge_pair, vocab)
        
        return merged_id

    def tokenize(self, text):
        
        final_tokens = {}
        for count, special_token in enumerate(sorted(self.special_tokens, key=len, reverse=True)):
            final_tokens[f" {count}123"] = special_token
            text = text.replace(special_token, f" {count}123")

        ####################################

        return [final_tokens.get(token_id, token_id) for token_id in re.findall(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", text)]

    ####################################
    
    def encode(self, text):
        
        final_ids = []
        for token_id in self.tokenize(text):
            if token_id not in self.special_tokens:
                final_ids.extend(self.apply_BPE_merges(token_id, self.merges, self.vocabulary))
            else:
                encoded_token = token_id.encode('utf-8')
                final_ids.extend([self.inverse_vocab[encoded_token]])

        return final_ids
    
    ####################################
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        
        iterable_buffer = ""
        for string in iterable:
            
            iterable_buffer = iterable_buffer + string
            current_lines = iterable_buffer.split('\n')

            ####################################
            
            split_buffer = iterable_buffer.split('\n')[:-1]
            for current_line in split_buffer:
                yield from self.encode(current_line + "\n")

            ####################################

            iterable_buffer = current_lines[-1] 

        ####################################
        
        if len(iterable_buffer) == 0:
            yield from self.encode(iterable_buffer)

    ####################################

    def decode(self, tokens: list[int]) -> str:
        
        return b''.join([self.vocabulary.get(id, '�') for id in tokens]).decode('utf-8', errors='replace')