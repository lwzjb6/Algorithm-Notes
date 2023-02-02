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

### 6294. 最大价值和与最小价值和的差值
`n = 6, edges = [[0,1],[1,2],[1,3],[3,4],[3,5]], price = [9,8,7,6,10,5], ans = 24`

给一个包含`n`个节点的无向图，`edges`给出边，`price[i]`表示节点`i`的价值，可以选择树中任意一个节点作为根节点`root`,找到最大价值的路径。返回路径的最大价值 - 最小价值（根节点的值或者叶子节点的值）。

#### 树形DP 
如何不用以每个节点作为根节点遍历整个图呢？
不要想着整条路径是从一个根节点出发直到叶子节点为止的一条线。
而是假定任意一个根节点，整条路径是以树中其中某个节点为根节点的一个分岔，即：路径=当前根节点+子树1+子树2 [关键点]

```c++
class Solution {
public:
    typedef long long ll;
    vector<vector<int>>g;
    ll ans = 0;
    pair<ll,ll> DFS(int x, int fa, vector<int>& price){
        ll p = price[x], max_s1 = p, max_s2 = 0; // 带叶子节点的最大值，不带叶子节点的最大值
        // 遍历所有的子树
        for(auto y : g[x]) {
            if(y != fa) {
                auto [s1, s2] = DFS(y, x, price);
                // 核心思路： 当前子树返回带叶子节点的最大值为s1, 少一个叶子节点的最大值为s2
                //           当前节点x之前的所有子树中带叶子节点的最大值为max_s1, 少一个叶子节点的最大值为max_s2
                // 按照题意：必须少一个端点
                ans = max(ans, max(max_s1 + s2, max_s2 + s1)); 
               

                max_s1 = max(max_s1, s1 + p); // 既然已经走到这里，说明当前节点x有子树，不是叶子结点，所以都可以加p
                max_s2 = max(max_s2, s2 + p);
            } 
        }
        return {max_s1, max_s2}; // 当前节点维护的信息，供其父节点看是否选择这条子链
    }
    long long maxOutput(int n, vector<vector<int>>& edges, vector<int>& price) {
        g = vector<vector<int>>(n);
        // 建图
        for(auto x : edges) {
            int a = x[0], b =x[1];
            g[a].push_back(b);
            g[b].push_back(a);
        }
        
        DFS(0, - 1, price);
        return ans;
    }
};
```

**时间复杂度：$O(n)$** 因为每个节点遍历了一次。

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

