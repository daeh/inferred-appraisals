
# %%

import numpy as np
import copy
import pprint
import json
from pathlib import Path


stim_rand_face_m = ["244_1", "275_1", "286_1", "288_1"]
stim_rand_face_f = ["250_1", "271_1", "272_1", "283_1"]

stim_base_m = [{ "stimulus": stimid, "pronoun": "this person", "desc": "" } for stimid in stim_rand_face_m]
stim_base_f = [{ "stimulus": stimid, "pronoun": "this person", "desc": "" } for stimid in stim_rand_face_f]

# stim_base = [
# 	{ "stimulus": "272_2", "pronoun": "he", "desc": "Software engineer at Google" }, ##M ##0
# 	{ "stimulus": "278_2", "pronoun": "she", "desc": "Software engineer at Google" }, ##F ##1

# 	{ "stimulus": "254_2", "pronoun": "he",  "desc": "Janitor at an elementary school" }, ##M ##2
# 	{ "stimulus": "249_2", "pronoun": "she",  "desc": "Retired waitress" }, ##F ##3

# 	{ "stimulus": "240_1", "pronoun": "he",  "desc": "Pursuing a career in rap music" }, ##M ##4
# 	{ "stimulus": "285_2", "pronoun": "she",  "desc": "Design student, wants to be fashion designer" }, ##F ##5

# 	{ "stimulus": "281_1", "pronoun": "he",  "desc": "Executive of a petroleum energy company" }, ##M ##6
# 	{ "stimulus": "246_1", "pronoun": "she",  "desc": "Corporate lawyer" }, ##F ##7

# 	{ "stimulus": "255_2", "pronoun": "he",  "desc": "CEO of a global health non-profit" }, ##M ##8
# 	{ "stimulus": "276_1", "pronoun": "she",  "desc": "Doctor, volunteering in South Africa with 'Doctors Without Borders'" }, ##F ##9

# 	{ "stimulus": "285_1", "pronoun": "he", "desc": "Investment analyst at a hedge fund" }, ##M ##10
# 	{ "stimulus": "280_1", "pronoun": "she",  "desc": "Stock broker at an investment firm" }, ##F ##11

# 	{ "stimulus": "239_1", "pronoun": "he", "desc": "Hospital nurse" }, ##M ##12
# 	{ "stimulus": "270_1", "pronoun": "she",  "desc": "High school english teacher" }, ##F ##13

# 	{ "stimulus": "269_2", "pronoun": "he",  "desc": "Boxing coach" }, ##M ##14
# 	{ "stimulus": "256_2", "pronoun": "she",  "desc": "Swimming coach" }, ##F ##15

# 	{ "stimulus": "266_2", "pronoun": "he", "desc": "City council member, about to start campaigning for State Senate" }, ##M ##16
# 	{ "stimulus": "263_2", "pronoun": "she", "desc": "Child therapist and special education counselor" }, ##F ##17
	
# 	{ "stimulus": "263_1", "pronoun": "he", "desc": "Police officer" }, ##M ##18
# 	{ "stimulus": "247_1", "pronoun": "she", "desc": "Operates an elderly care center" }, ##F ##19
# ]

# ##



pots_full = [2,11,25,46,77,124,194,299,457,694,1049,1582,2381,3580,5378,8075,12121,18190,27293,40948,61430,92153,138238,207365]
pots_selected = [124, 694, 1582, 5378, 12121, 27293, 61430, 138238]


continue_randomization = True
loop_count = 0
fail_count1 = 0
fail_count2 = 0


outcomes = [
{"decisionThis": "Split", "decisionOther": "Split"}, 
{"decisionThis": "Split", "decisionOther": "Stole"}, 
{"decisionThis": "Stole", "decisionOther": "Split"}, 
{"decisionThis": "Stole", "decisionOther": "Stole"}, ]

# stim_selector_set = [
# 			np.array([True, True, True, True, False, False, False, False, False, False]),
# 			# np.array([False, True, True, True, True, False, False, False, False, False]),
# 			np.array([False, False, True, True, True, True, False, False, False, False]),
# 			# np.array([False, False, False, True, True, True, True, False, False, False]),
# 			np.array([False, False, False, False, True, True, True, True, False, False]),
# 			# np.array([False, False, False, False, False, True, True, True, True, False]),
# 			np.array([False, False, False, False, False, False, True, True, True, True]),
# 			# np.array([True, False, False, False, False, False, False, True, True, True]),
# 			np.array([True, True, False, False, False, False, False, False, True, True]),
# 			# np.array([True, True, True, False, False, False, False, False, False, True]),
# 		]

outcome_idx_map_set = [
	[0,1,2,3],
	[3,0,1,2],
	[2,3,0,1],
	[1,2,3,0],
]



while continue_randomization and loop_count < 1e6:
	if loop_count == 0:
		print('starting...')

	if loop_count % 10000 == 0:
		print(f"{loop_count} :: {fail_count1}, {fail_count2}")

	loop_count += 1

	stim_set = list()

	# for i_stim_set in [1,2]:
		
	# Mlist = np.array(copy.copy(stim_base[::2]))
	# Flist = np.array(copy.copy(stim_base[1::2]))

	# assert len(Flist) == len(Flist)
	# assert all M and all F

	# np.random.shuffle(Mlist)
	# np.random.shuffle(Flist)

	

	for pot_map_idx in range(len(pots_selected)):
		pots = np.roll(pots_selected, pot_map_idx)
		
		for outcome_idx_map in outcome_idx_map_set:
			
			# for i_selector,selector in enumerate(stim_selector_set):

			# assert selector.sum() == 4

			Msel = copy.deepcopy(stim_base_m)
			Fsel = copy.deepcopy(stim_base_f)

			for i_outcome,outcome_idx in enumerate(outcome_idx_map):
				outcome = outcomes[outcome_idx]
				Msel[i_outcome].update(outcome)
				Fsel[i_outcome].update(outcome)
			
			stim_list = [*Msel, *Fsel]
			
			for i_pot,pot in enumerate(pots):
				stim_list[i_pot]['pot'] = int(pot)

			stim_set.append( stim_list )


	counts = dict()
	blank = np.full((len(stim_set), len(stim_list)), None, dtype=np.object)
	stim_mat = blank.copy()
	counts['gender'] = blank.copy()
	counts['stim'] = blank.copy()
	counts['outcome'] = blank.copy()
	counts['pot'] = np.full_like(blank, 0, int)
	counts['genderXoutcome'] = blank.copy()
	counts['stimXoutcome'] = blank.copy()
	counts['stimXoutcomeXpot'] = blank.copy()

	for i_list,stimlist in enumerate(stim_set):
		for i_stim,stim in enumerate(stimlist):
			a1 = {"Split": "C", "Stole": "D"}[stim["decisionThis"]]
			a2 = {"Split": "C", "Stole": "D"}[stim["decisionOther"]]
			
			stim_mat[i_list,i_stim] = stim
			counts['gender'][i_list,i_stim] = stim['pronoun']
			counts['stim'][i_list,i_stim] = stim['stimulus']
			counts['outcome'][i_list,i_stim] = f'{a1}{a2}'
			counts['pot'][i_list,i_stim] = stim['pot']
			counts['genderXoutcome'][i_list,i_stim] = f"{stim['pronoun']}{a1}{a2}"
			counts['stimXoutcome'][i_list,i_stim] = f"{stim['stimulus']}{a1}{a2}"
			counts['stimXoutcomeXpot'][i_list,i_stim] = f"{stim['stimulus']}{a1}{a2}{stim['pot']}"

		crit1 = True
		if np.sum( counts['stim'][i_list,:] == '278_2' ) + np.sum( counts['stim'][i_list,:] == '272_2' ) == 2:
			fail_count1 += 1
			crit1 = False
			break
			

	
	if crit1:
		crit2 = True
		fa1a2, cross_counts = np.unique(counts['stimXoutcome'], return_counts=True)
		if len(np.unique( cross_counts )) > 1:
			fail_count2 += 1
			crit2 = False
			

	if crit1 and crit2:
		continue_randomization = False


print(f'done :: {loop_count}')
# %%


for i_list,stimlist in enumerate(stim_set):
			
	### balance of gender x outcome in each list
	ga1a2, cross_counts = np.unique(counts['genderXoutcome'][i_list,:], return_counts=True)
	assert len(np.unique( cross_counts )) == 1
	# assert np.unique( cross_counts )[0] == 1
	
	### make sure any given face only appears once in a list
	# face, cross_counts = np.unique(counts['stim'][i_list,:], return_counts=True)
	# assert np.unique( cross_counts )[0] == 1
	
	### make sure each pot appears once in a list
	potstemp, cross_counts = np.unique(counts['pot'][i_list,:], return_counts=True)
	assert np.unique( cross_counts )[0] == 1
	
	### randomize until these two stim don't co-occur
	assert np.sum( counts['stim'][i_list,:] == '278_2' ) + np.sum( counts['stim'][i_list,:] == '272_2' ) < 2

### balanced gender x outcome in each list
ga1a2, cross_counts = np.unique(counts['genderXoutcome'], return_counts=True)
assert len(np.unique( cross_counts )) == 1

### balanced face x outcome in each list
fa1a2, cross_counts = np.unique(counts['stimXoutcome'], return_counts=True)
assert len(np.unique( cross_counts )) == 1

### balanced face x outcome in each list
fa1a2pot, cross_counts = np.unique(counts['stimXoutcomeXpot'], return_counts=True)
assert len(np.unique( cross_counts )) == 1
	
### make sure faces appear an equal number of times across set
face, cross_counts = np.unique(counts['stim'], return_counts=True)
assert len(np.unique( cross_counts )) == 1


counts['index'] = np.full_like(blank, -1, dtype=np.int)
stim_list = list()
for i_stimid,stimid in enumerate(fa1a2pot):
	counts['index'][ counts['stimXoutcomeXpot'] == stimid ] = i_stimid
	stim_temp = stim_mat[ counts['stimXoutcomeXpot'] == stimid ][0]
	stim_temp['index'] = i_stimid
	stim_list.append( copy.deepcopy(stim_temp) )
assert np.sum(counts['index'] < 0) == 0
	

print(f"nConditions = {counts['index'].shape[0]}")
print(f'nStim = {len(stim_list)}')

# %%


with open(Path.home() / 'Desktop' / 'conditions_11.json', 'w', encoding='utf-8') as f:
	json.dump(counts['index'].tolist(), f, ensure_ascii=False, indent='\t', separators=(',', ':'))

### print stim dict
with open(Path.home() / 'Desktop' / 'stim_11.json', 'w', encoding='utf-8') as f:
	json.dump(stim_list, f, ensure_ascii=False, indent='\t')

# with open(Path.home() / 'Desktop' / 'stim.json', 'w') as f:
#     json.dump(stim_list, f)


#%%
print(blank[(4,4)])

"""
balanced M/F
balanced C/D
same face not included twice
every _ number of responses balances the stim
0 and 1 not included in same set


divide into M F lists, shuffle
take 4 from each list
assign C to first 2, D to second 2

pot size...
"""