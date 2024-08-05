
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:

      # Create a list which we eill iterate over
      left = 0
      max_len = 0
      char_set = set()
      
      for right in range(len(s)):
        while s[right] in char_set:
          char_set.remove(s[left])
          left += 1

        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)

      return max_len
    
    def countCompleteSubarrays(self, nums):

      n = len(nums)
      m = len(set(nums))
      count = 0

      for i in range(n):
        sub_set = set()

        for j in range(i, n):
          sub_set.add(nums[j])

          if len(sub_set) == m:
            count += 1
      return count
          
test = Solution()
nums = [5,5,5,5]
print(test.countCompleteSubarrays(nums))