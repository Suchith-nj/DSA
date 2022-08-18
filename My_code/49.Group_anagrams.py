'''
Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, 
typically using all the original letters exactly once.

Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]'''

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        res = defaultdict(list)
        
        for string in strs:
            count = [0]* 26 # results in [0,0,0,0, ....0]
            
            for letter in string:
                count[ord(letter)-ord("a")] +=1
            
            res[tuple(count)].append(string)        #key of dic can not be list hence tuple
        return res.values()
        