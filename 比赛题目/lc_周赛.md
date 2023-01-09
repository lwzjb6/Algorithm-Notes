<font face="楷体" size = 3>

### 6269. 到目标字符串的最短距离
在`words`中从索引`startIndex`开始找到与`target`相等的字符的最近的距离。

`words = ["hello","i","am","leetcode","hello"], target = "hello", startIndex = 1, ans = 1`

```c++
class Solution {
public:
    int closetTarget(vector<string>& words, string target, int startIndex) {
        int n = words.size();
        int ans = n;
        for (int i = 0; i < n; i++) {
            if (words[i] == target) {
                int len = abs(i - startIndex);
                ans = min(ans, min(len, n - len));
            }
        }
        return ans == n ? -1 : ans;
    }
};
```
---
### 6270. 每种字符至少取 K 个
字符串`s`由`a, b, c`构成，每次可以选择取走 `s` 最左侧或者最右侧的字符。
问三种字符至少取走K个所需要的最小次数。

#### 思路1：二分
先从右向左预处理出每个位置三个字符的数量。时间复杂度$O(n)$
然后从`0~n`枚举左边元素的数量, 用变量`na,nb,nc`维护左区间三个字符的数量。
固定左边，用二分找到右端点，左区间的右端点。
总时间复杂度$O(nlogn)$

#### 思路2：滑动窗口
两边找最短的区间使得所有字符的数量都大于等于K个，等价于在中间找一个最长的区间，使得其中所有字符的数量小于$n_i - K$个

当字符小于时，不断扩展右区间，当字符大于时，拓展左区间，找到最大的窗口。
时间复杂度$O(n)$
```c++
class Solution {
public:
    int takeCharacters(string s, int k) {
        int n = s.size();
        int na = count(s.begin(), s.end(), 'a');
        int nb = count(s.begin(), s.end(), 'b');
        int nc = count(s.begin(), s.end(), 'c');
        if (na < k || nb < k || nc < k) return -1;
        na -= k, nb -= k, nc -= k;
        // 找到最长的窗口，其中每个字符的数量不超过ni
        vector<int>nums(3, 0), up={na, nb, nc};
        int maxn = 0;
        for(int l = 0, r = 0; r < s.size(); r++) {
            int c = s[r] - 'a';
            nums[c] ++;
            while (nums[c] > up[c]) {
                nums[s[l] - 'a']--;
                l++;
            }
            maxn = max(maxn, r - l + 1);
        }
        return n - maxn;
    }
};
```
---
### 6271. 礼盒的最大甜蜜度
`price = [13,5,1,8,21,2], k = 3, ans = 8`
（选择 `5, 13, 21`）

从`price`中选`k`个数, 使得它们彼此之间的**最小的差值最大。**

#### 二分
二分最大的差值(答案), 然后判断以当前差值能否从原数组中挑选出K个
```c++
class Solution {
public:
    // 判断以gap为间隔，在a中能否选出k个数
    bool check(vector<int>& a, int gap, int k) {
        int cnt = 1, pre = a[0];
        for (int i = 1; i < a.size(); i++) {
            if(a[i] - pre >= gap) {
                cnt++;
                pre = a[i];
            }
        }
        if (cnt >= k) return 1;
        else return 0;
    }
    int maximumTastiness(vector<int>& p, int k) {
        sort(p.begin(), p.end());
        int l = 0, r =  1e9;
        while(l < r) { // 左区间的右端点
            int mid = (l + r + 1) >> 1;
            if (check(p, mid, k)) l = mid;
            else r = mid - 1;
        }
        return l;
    }
};
```
---
### 6272. 好分区的数目
将数组中的数分成两个分区，如果每个分区的元素和大于等于K，则认为是一个好分区，求好分区的数量。注意：`[[1],[2]]` 和`[[2],[1]]`算不同的方案。
`nums = [1,2,3,4], k = 4, ans = 6`
好分区的情况是 `([1,2,3], [4]), ([1,3], [2,4]), ([1,4], [2,3]), ([2,3], [1,4]), ([2,4], [1,3]) 和 ([4], [1,2,3])` 。

#### 01背包求方案数
问题等价于在数组中挑选几个数组成A分区（剩下的数去B分区），每个数选或者不选，一共$2^n$种方案，不合法的方案数为`2` * `sum(dp[j], for j < k)   dp[j]表示`选择出的数字和为j的方案数。前面`2`的意思是B分区同理。最终合法的方案数就是两者的差值。
```c++
class Solution {
public:
    typedef long long ll;
    const int mod = 1e9 + 7;
    int countPartitions(vector<int>& nums, int k) {
        ll sum  = 0;
        for (auto x : nums) sum += x;
        if (sum < 2 * k) return 0;
        vector<int>f(k, 0);
        int ans = 1;
        f[0] = 1;
        for(int i = 0; i < nums.size(); i++) {
            ans = (ans << 1) % mod; // 所有可能的情况是2^n;
            for(int j = k - 1; j >= nums[i]; j--) {
                f[j] = (f[j] + f[j - nums[i]]) % mod;
            }
        }
        // 因为上面的情况仅考虑将元素选入A分区的情况数，所以要乘2;
        // f[0] = 1什么也不选也要考虑，相当于A分区一个数都没有
        for (auto x : f) {
            ans = (ans - 2 * x % mod + mod) % mod;
        }
        return ans;
    }
};
```
---