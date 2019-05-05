#!/usr/bin/env bash
# sentence_text.txt
# --layers=-5
#  --input_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/data/cf/truncated_onto_sentences.txt \
#  --output_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/contextualizers/bert_base_cased/cf/coreference_resolution.hdf5 \
python extract_document_feature.py \
  --input_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/data/cf/truncated_onto_sentences.txt \
  --output_file /home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/contextualizers/qqp/cf/coreference_resolution.hdf5 \
  --bert_model output/QQP \
  --max_seq_length 512 \
  --doc_stride 64
