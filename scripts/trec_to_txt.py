from collections import defaultdict


in_file_path="/data/user_data/jmcoelho/datasets/marco/documents/gen6.bm25.run.trec"
out_file_path="/data/user_data/jmcoelho/datasets/marco/documents/gen6.bm25.top100.run.trec"
out_out_file_path="/data/user_data/jmcoelho/datasets/marco/documents/gen6.bm25.top.100run.txt"

# with open(in_file_path, 'r') as h,  open(out_file_path, 'w') as o:
#     for line in h:
#         qid, _, did, pos, score, _ = line.strip().split()

#         if int(pos) > 100:
#             continue
        
#         else:
#             o.write(f"{qid} {did} {score}\n")


# counter = defaultdict(int)

# with open(out_file_path, 'r') as h:
#     for line in h:
#         qid, did, score = line.strip().split()

#         counter[qid] += 1


# for qid in counter:
#     if counter[qid] < 100:
#         counter[qid] = False
#     else:
#         counter[qid] = True


# for i in range(0, len(counter), 4):
#     pair_0 = counter[str(i)]
#     pair_1 = counter[str(i+1)]        
#     pair_2 = counter[str(i+2)]        
#     pair_3 = counter[str(i+3)]        
#     if not pair_0 or not pair_1 or not pair_2 or not pair_3:
#         counter[str(i)] = False
#         counter[str(i+1)] = False
#         counter[str(i+2)] = False
#         counter[str(i+3)] = False
        
# with open(out_file_path, 'r') as h, open(out_out_file_path, 'w') as o:
#     for line in h:
#         qid, did, score = line.strip().split()

#         if counter[qid] is False:
#             continue
            
#         o.write(f"{qid} {did} {score}\n")


qids = set()
with open(out_out_file_path, 'r') as o:

    for line in o:
        qid, did, s = line.strip().split()
        qids.add(qid)

for i in range(0, 200000, 4):
    if (str(i) not in qids) != (str(i+1) not in qids) != (str(i+2) not in qids) != (str(i+3) not in qids):
        print(str(i), str(i+1), str(i+2), str(i+3))

