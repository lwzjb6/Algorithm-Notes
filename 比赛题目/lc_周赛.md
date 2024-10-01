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

### 6292. 子矩阵元素加 1
假定原数组全部为`0`，在子矩阵`l1,r1,l2,r2`添加1，问多次修改后，数组为什么？

思路：二维差分+二维前缀和的模板题
难点在于：从索引1开始做比较方便，但是结果返回的是从索引0
解决办法：先按索引1做，最后拷贝一下。

```c++
class Solution {
public:
    vector<vector<int>> rangeAddQueries(int n, vector<vector<int>>& queries) {
        vector<vector<int>>d(n + 2, vector<int>(n + 2, 0)); // 因为原数组全为2，所以其差分序列仍旧为0
        for(auto x : queries) {
            int l1 = x[0] + 1, r1 = x[1] + 1, l2 = x[2] + 1, r2 = x[3] + 1; // 索引从1开始，所以都加1
            d[l1][r1] += 1, d[l2 + 1][r1] -= 1, d[l1][r2 + 1] -= 1, d[l2 + 1][r2 + 1] += 1;
        }
        // 求二维前缀和
        for(int i = 1; i <= n; i++) 
            for(int j = 1; j <= n; j++) 
                d[i][j] = d[i - 1][j] + d[i][j - 1] - d[i - 1][j - 1] + d[i][j];
            
        // 转移一下
        vector<vector<int>>ans(n, vector<int>(n, 0));
         for(int i = 0; i < n; i++) 
            for(int j = 0; j < n; j++)
                ans[i][j] = d[i + 1][j + 1];
        return ans; 
    }
};
```
---

### 6293. 统计好子数组的数目
一个子数组 `arr` 如果有 **至少 `k `对**下标 `(i, j)` 满足 `i < j` 且 `arr[i] == arr[j]` ，那么称它是一个好子数组。

`nums = [3,1,4,3,2,2,4], k = 2, ans = 4`
```c++
总共有 4 个不同的好子数组：
- [3,1,4,3,2,2] 有 2 对。
- [3,1,4,3,2,2,4] 有 3 对。
- [1,4,3,2,2,4] 有 2 对。
- [4,3,2,2,4] 有 2 对。
```

#### 思路：滑动窗口
题目的基本性质：
如果一个子数组是好数组，那么往其中加任何数都是好数组。

`r`指针不断向右，直到当前数组满足要求，当满足要求后，尝试移动左指针，找到r固定后最小的滑动窗口，此时`0~l`之间的任何一个数与`r`构成的子数组均满足要求。因此答案加`l+1`[本质上是统计每个以r结尾的满足要求的子数组的个数]

快速统计加入一个数后，满足要求的对数。
如果新加的数是`x`, 其之前出现了`n`次，加入后出现了`n+1`次
因此新增的对数为：$C_{n + 1}^2 - C_{n}^2 = n $. 也可以简单理解，新增的数会和之前的每个数组成一对，所以新增了n次

同理：如果新减的数是`y`, 其之前出现了`n`次，减掉后出现了`n-1`次，对数减少了`n - 1`对


```c++
class Solution {
public:
    typedef long long ll;
    long long countGood(vector<int>& nums, int k) {
        int n = nums.size();
        ll ans = 0, res = 0; // res统计当前子数组内的好数组的个数
        unordered_map<int, int>hx; //统计每个数出现的次数
        for(int l = 0, r = 0; r < n; r++) {
            res += hx[nums[r]]; // 好数组的对数
            hx[nums[r]] ++;
            while(res - (hx[nums[l]] - 1) >= k) { // 尝试移动左指针,移动后仍然满足条件
                hx[nums[l]]--;
                res -= (hx[nums[l]]);
                l++;
            }
            // 移动完成后，以r指针作为结尾的并且满足要求的子数组有 l + 1个
            if(res >= k) ans += (l + 1);
        }
        return ans;
    }
};
```
---

### 2547. 拆分数组的最小代价

将数组拆分成一些非空子数组。拆分的**代价** 是每个子数组中的 **重要性** 之和。
令 `trimmed(subarray)` 作为子数组的一个特征，其中所有仅出现一次的数字将会被移除。
例如，`trimmed([3,1,2,4,3,4]) = [3,4,3,4]`。
数组的 重要性 定义为 `k + trimmed(subarray).length` 。
找出并返回拆分 `nums` 的所有可行方案中的最小代价。

`nums = [1,2,1,2,1,3,3], k = 2.   ans = 8`
```c++
将 nums 拆分成两个子数组：[1,2], [1,2,1,3,3]
[1,2] 的重要性是 2 + (0) = 2 。
[1,2,1,3,3] 的重要性是 2 + (2 + 2) = 6 。
拆分的代价是 2 + 6 = 8 
```
#### DP

遇到这种拆分子数组，不知道具体个数的情况通常用`DP`
`f[i]`表示以`i`结尾的元素之前的拆分方案的最优解
`f[i] = min(f[i], f[j] + ....)`
用两重循环:
```python
for i in range(n)：
    for j in range(i):
        f[i] = ...
```

```c++
class Solution {
public:
    int trim[1010][1010]; 
    int minCost(vector<int>& nums, int k) {
        int n = nums.size();
        // 预处理出trim[i][j]
        for(int i = 0; i < n; i ++) {
            unordered_map<int, int>hx;
            int t = 0;
            for(int j = i; j < n; j++) {
                int x = nums[j];
                hx[x]++;
                if(hx[x] == 2) t += 2;
                else if(hx[x] > 2) t++;
                trim[i][j] = t; // 转化为出现一次的不考虑，其次数字加起来
            }
        }
        // DP
        vector<int>dp(n + 1, INT_MAX); // `dp[i]`表示以`i`结尾的元素之前的拆分方案的最优解
        dp[0] = 0; // 边界 dp[i] 对应元素nums[i - 1];
        for(int i = 1; i <= n; i++) {
            for(int j = 0; j < i; j++) {
                dp[i] = min(dp[i], dp[j] + k + trim[j][i - 1]); // dp的索引和nums的索引错一个
            }
        }
        return dp[n];
    }
};
```
---


### 6340. 统计上升四元组
返回上升四元组的数目。
如果一个四元组 `(i, j, k, l)` 满足以下条件，我们称它是上升的：
`0 <= i <j < k < l < n` 且 `nums[i] < nums[k] < nums[j] < nums[l]`
注意中间两个是反的，不是常规理解的上升。
`nums = [1,3,2,4,5] ans = 2`
`[1 3 2 4] 和 [1 3 2 5]`

**提示：**
(1): 只考虑 `i, j, k, l`中的`j, k`
(2): 
在`k`右侧的比`nums[j]`大的元素个数，记作 `great[k][nums[j]]`;
在`j `左侧的比`nums[k]`小的元素个数，记作 `less[j][nums[k]]`。

```c++
class Solution {
public:
    static const int N = 4010;
    int greater[N][N];
    using ll = long long;
    long long countQuadruplets(vector<int>& nums) {
        int n = nums.size();
        ll ans = 0;
        // greater[k][x]: 在k右侧比x大的数的个数
        for(int k = n - 2; k >= 2; k--) {
            memcpy(greater[k], greater[k + 1], sizeof(greater[k + 1])); // 利用之前的结果
            for(int x = nums[k + 1] - 1; x > 0; x--) {
                greater[k][x] ++;
            }
        }
        vector<int>less(n + 1, 0);
        // less[x]: 在当前j左侧比x小的数的个数
        for(int j = 1; j < n - 2; j++) {
            for(int x = nums[j - 1] + 1; x <= n; x++) {
                less[x] ++;
            }
            // 枚举所有可能的j,k组合，其合法的个数为greater[k][nums[j]] * less[j][nums[k]]
            for(int k = j + 1; k <= n - 2; k++) {
                if(nums[k] < nums[j]) { // 满足 j,k的要求
                    ans += greater[k][nums[j]] * less[nums[k]];
                }
            }
        }
        return ans;
    }
};
```
---

### 6355. 统计公平数对的数目
问数组中有多少对`i,j, i < j`,使得 `lower < nums[i] + nums[j] < upper`
`ums = [0,1,7,4,4,5], lower = 3, upper = 6`

#### 二分

首先排序，对于每个数字`nums[i]`, 在它之后用`lower_bound`找出大于等于`（lower - nums[i]）`的第一个位置，记为指针`pos1`, 再用`upper_bound`找出大于`upper - nums[i]`的第一个位置，记为指针`pos2`;
`pos1`表示其之后的数(包括本位) 满足`lower`的要求
`pos2`表示其之后的数(包括本位)不满足`upper`的要求
因此，`[pos1,pos2)`之前的数就是符合要求的个数，个数为：`pos2 - pos1`;
```c++
class Solution {
public:
    using ll = long long;
    long long countFairPairs(vector<int>& nums, int lower, int upper) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        ll ans = 0;
        for(int i = 0; i < n; i++) {
            auto start = nums.begin() + i + 1;
            int pos1 = lower_bound(start, nums.end(), lower - nums[i]) - nums.begin();
            int pos2 = upper_bound(start, nums.end(), upper - nums[i]) - nums.begin();
            ans += pos2 - pos1;
        }
        return ans;
    }
};
```
---

### 6356. 子字符串异或查询
给一个字符串`s`和一组查询`{[a, b]}`, 找出`s`的最短子串，其十进制表示等于`a ^ b`
`s = "101101", queries = [[0,5],[1,2]]`

数据范围：
```c++
N = s.size() == 1e4
M = queries.size() == 1e5
```
对于每组询问，如果采取滑动窗口的方法或者`KMP`的话，其每组查询时间复杂度为`O(N)`总时间复杂度为`1e9`, 会超时，因此必须在`O(logn)`时间内完成每组的查询

#### 预处理
可以发现，不同于之前常规的字符串匹配问题，每次`s, p`都是不同的，本题是对于一个`s`，有多组`p`，因此可以预处理出`s`的相关信息。
对于`s`中的每个索引，可以找出从当前索引往后的`30`位(预处理 `30` 位的原因是因为数据量最大是 `10^9`，对应二进制最多 `30` 位)，因此时间复杂度O(30 * 1e4)就可以处理完所有的情况。

```c++
class Solution {
public:
    typedef pair<int, int> pii;
    vector<vector<int>> substringXorQueries(string s, vector<vector<int>>& queries) {
        unordered_map<int, pii>hx;
        int n = s.size();
        // 预处理
        for(int l = 0; l < n; l++) {
            int x = 0;
            for(int r = l;  r < min(l + 30, n); r++) {
                x = x * 2 + (s[r] - '0'); // 当前字串构成的数字
                if(!hx.count(x)) hx[x] = {l, r};
                else { // 看当前的是不是更短
                    auto [a, b] = hx[x];
                    if(b - a > r - l) hx[x] = {l, r}; 
                } 
            }
        }
        vector<vector<int>> ans;
        for(auto x : queries) {
            int num = x[0] ^ x[1];
            if(hx.count(num)) ans.push_back({hx[num].first, hx[num].second});
            else ans.push_back({-1, -1});
        }
        return ans;
    }
};
```
---

### 1124. 表现良好的最长时间段
大于8表示好，问好的时间大于不好的时间的最长时间段
`hours = [9,9,6,0,6,6,9], ans = 3, [9,9,6]`
等价问题:  假设一个数组由`1，-1`，构成，问区间分数和大于`0`的最长区间长度

#### 前缀和 + 哈希表
```c++
class Solution {
public:
    int longestWPI(vector<int>& hours) {
        int n = hours.size();
        int sum = 0; // 记录前缀和
        int ans = 0;
        unordered_map<int, int>hx; // 统计某个数出现的索引
        for(int i = 0; i < hours.size(); i++) {
            sum += (hours[i] > 8) ? 1 : -1; // 维护前缀和
            if(sum > 0) {
                ans = max(ans, i + 1); // 从头到现在都可以
            }
            else {  // 找到前面距离最远的j, sum[j] < sum[i], 其中sum[j] = sum[i] - 1是距离i最远的
                if(hx.count(sum - 1)) ans = max(ans, i - hx[sum - 1]);
            }
            if(!hx.count(sum)) hx[sum] = i;
        }
        return ans;
    }
};
```
---

### 6365. 将整数减少到零需要的最少操作数
给一个整数`n`,两种操作：
(1)加上`2`的`n`次方
(2)减去`2`的`n`次方
问将整数减少到零需要的最少操作数

#### 贪心
`n` 最低位为`0`则右移，不增加操作数
`n` 的二进制末尾有连续两个 `1` 以上做 `+` 操作
`n` 的二进制末尾只有一个 `1` 做 `-` 操作

```c++
class Solution {
public:
    int minOperations(int n) {
        int step = 0;
        while(n) {
            if((n & 3) == 3) {
                n += 1;
                step++;
            } 
            else if((n & 1) == 1){
                n -= 1;
                step++;
            }
            n >>= 1;
        }
        return step;
    }
};
```
---


### 2611. 老鼠和奶酪
`reward1 = [1,1,3,4], reward2 = [4,4,1,1], k = 2`
`ans = 15`
有两只老鼠，每个奶酪给它后的奖励是`reward`, 问第一只老鼠恰好吃掉 `k` 块奶酪的情况下，最大得分为多少。

#### 思路1：DP
```c++
class Solution {
public:
    int miceAndCheese(vector<int>& r1, vector<int>& r2, int k) {
        int n = r1.size();
        int f[n + 1][k + 1];
        memset(f, -1, sizeof(f));
        f[0][0] = 0;
        for(int i = 1; i <= n; i++) f[i][0] = f[i - 1][0] + r2[i - 1];
        for(int i = 1; i <= n; i++) {
            for(int j = 1; j <= k; j++) {
                f[i][j] = max(f[i - 1][j] + r2[i - 1], f[i - 1][j - 1] + r1[i - 1]);
            }
        }
        return f[n][k];
    }
};
```

#### 思路2：贪心
先全部给第二只老鼠，之后计算`reward1[i] - reward2[i]`然后贪心选增量最大的`k`个即可
```c++
class Solution {
public:
    int miceAndCheese(vector<int>& r1, vector<int>& r2, int k) {
        for(int i = 0; i < r1.size(); i++) r1[i] -= r2[i];
        nth_element(r1.begin(), r1.end() - k, r1.end()); // 快速选择
        return accumulate(r2.begin(), r2.end(), 0) + accumulate(r1.end() - k, r1.end(), 0);
    }
};
```
---

### 3296. 移山所需的最少秒数
山的高度降低 x，需要花费 `workerTimes[i] + workerTimes[i] * 2 + ... + workerTimes[i] * x` 秒。

```
mountainHeight = 4, workerTimes = [2,1,1]
ans = 3
```
#### 方法1： 最小堆
```python
class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        h = [(x, 1, x) for x in workerTimes]
        heapify(h)
        maxn = 0
        for _ in range(mountainHeight):
            num, cnt, base = heappop(h)
            maxn = max(maxn, num)
            heappush(h, (num + (cnt + 1) * base, cnt + 1, base))
        return maxn
```
时间复杂度：O(n)

#### 方法二： 二分
问题转化为：每个人最多花费m秒，看能否将山的高度降低mountainHeight，二分枚举m即可。
难点在于如何根据m计算出每个人能降低的高度x。

根据公式可以计算出：
对于某个人，其`worktime[i] = t`;
则 $x  = \frac{(-1 + \sqrt{1 + 4k})}{2} $, 其中`k = 2 * m/ t`

```python
class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        def check(m):
            res = 0
            for t in workerTimes:
                k = 2 * m / t
                x = (-1 + sqrt(1 + 4 * k)) // 2
                res += x
            return 1 if res >= mountainHeight else 0 

        max_t = max(workerTimes) # 假设每个人都是工作效率最低的那个人
        hi = (mountainHeight - 1) // len(workerTimes) + 1 # 求H/n的上界，即每个人平均需要降低的高度
        R = max_t * (1 + hi) * hi // 2
        return bisect_left(range(R), True, 1, key=check)
```

#### 3298. 统计重新排列后包含另一个字符串的子字符串数目 II

找出word1中有多少个字字符串，其重新排列后等于word2

```
word1 = "abcabc", word2 = "abc"
ans = 10
```

#### 滑动窗口找字串经典题目
```python
class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        hx = defaultdict(int) # 当访问的key不存在时，会返回0
        for x in word2:
            hx[x] += 1
        len1, len2 = len(word1), len(word2)
        l = r = cnt = res = 0
        while r < len1:
            c = word1[r]
            if hx[c] > 0:
                cnt += 1
            hx[c] -= 1
            while cnt == len2:
                res += (len1 - r)
                hx[word1[l]] += 1
                if hx[word1[l]] > 0: # 说明左指针右移让一个字符不够了
                    cnt -= 1
                l += 1
            r += 1
        return res
```

```python
class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        js = Counter(word2) # 记录Word2中每个单词出现的个数
        l = cnt = res = 0
        for c in word1:
            if js[c] > 0:
                cnt += 1
            js[c] -= 1
            while cnt == len(word2): # 找到了所有的
                if js[word1[l]] == 0:
                    cnt -= 1
                js[word1[l]] += 1
                l += 1
            res += l
        return res
```

####  3306：包含每个元音和 K 个辅音的子串数量 II
题意：统计包含5个元音至少一个并且包含k个辅音的字串的数量

转换：恰好型滑动窗口：转换成两个至少型滑动窗口
问题等价于如下两个问题：

- 每个元音字母至少出现一次，并且至少包含 k 个辅音字母的子串个数。记作 $f_k$
- 每个元音字母至少出现一次，并且至少包含 k+1 个辅音字母的子串个数。记作 $f_{k+1}$
​
结果为：$f_{k+1} - f_{k}$
```python
class Solution:
     # 统计5个元音至少出现一次，并且至少包含k个辅音的字串的个数
    def f(self, word: str, k: int) -> int:
        cnt1 = defaultdict(int) # 哈希表，统计元音的个数
        cnt2 = ans = l = 0
        for c in word:
            if c in 'aeiou':
                cnt1[c] += 1
            else :
                cnt2 += 1
            while len(cnt1) == 5 and cnt2 >= k:
                lc = word[l]
                if lc in 'aeiou':
                    cnt1[lc] -= 1
                    if cnt1[lc] == 0:
                        del cnt1[lc]
                else:
                    cnt2 -= 1
                l += 1
            ans += l   
        return ans     
    def countOfSubstrings(self, word: str, k: int) -> int:
        return self.f(word, k) - self.f(word, k + 1)
```

#### 3307. Find the K-th Character in String Game II
题意：初始是一个字符a, 然后经过操作 operations 后，返回第k个字符
其中：
operations[i] == 0: 直接复制前半部分
operations[i] == 1: 前半部分字符ASCII码+1

```
Input: k = 10, operations = [0,1,0,1]
Output: "b"

Appends "a" to "a", word becomes "aa".
Appends "bb" to "aa", word becomes "aabb".
Appends "aabb" to "aabb", word becomes "aabbaabb".
Appends "bbccbbcc" to "aabbaabb", word becomes "aabbaabbbbccbbcc".
```

#### 方法1：递归，分治，转化为更小的子问题
```python
class Solution:
    def kthCharacter(self, k: int, operations: List[int]) -> str:
        if len(operations) == 0: # 终止条件
            return 'a'
        n = len(operations)
        op = operations.pop() # 将操作缩短一个
        m = (1 << (n - 1))
        if k <= m: # 在左半边
            return self.kthCharacter(k, operations)
        else: # 在右半边
            res = self.kthCharacter(k - m, operations)
            res = (ord(res) - ord('a') + op) % 26
            return ascii_lowercase[res]
```

#### 方法2：迭代
```python
class Solution:
    def kthCharacter(self, k: int, operations: List[int]) -> str:
        n = len(operations)
        inc = 0 # 统计需要加的次数
        for i in range(n - 1, -1, -1):
            m = 1 << i
            if k > m:
                inc += operations[i]
                k -= m
        return ascii_lowercase[inc % 26]
```
