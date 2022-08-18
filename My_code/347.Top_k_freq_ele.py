'''
Given an integer array nums and an integer k, return the k most frequent elements. 
You may return the answer in any order.

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]'''

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        dic = {}
        for num in nums:
            dic[num] = dic.get(num,0)+1             #dic[num] = count
        
        bb = [[] for i in range (len(nums)+1)]      #CREATING BUCKET OF BUCKETS #OBSERVE SYNTAX
        
        for key, val in dic.items():
            bb[val].append(key)                     #bucket[count]= val
            
        result = []
        
        for x in range (len(bb)-1, -1, -1):     #Fetching from most freq elements
            for y in bb[x]:
                result.append(y)
                if len(result)==k:
                    return result