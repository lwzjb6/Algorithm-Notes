<font face= "楷体" size = 3>

<center><font face="楷体" size=6, color='red'> 最长上升子序列问题(LIS)
 </font> </center>


### 300. 最长递增子序列
题意：子序列：保持顺序不变，可以不连续

#### 思路1：普通DP
**时间复杂度$O(n^2)$**
```c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int>f(n, 1);
        int ans = 1;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++)
                if (nums[i] > nums[j]) 
                    f[i] = max(f[i], f[j] + 1);
            ans = max(ans, f[i]);
        }
        return ans;
    }
};
```
#### 思路2：模拟栈优化
遍历原始序列，维护一个单调的数组(栈)，之所以用数组是因为可以用`stl`的`lower_bound`
大致思路： 如果当前的元素`x`大于单调数组的最后一个元素，那么直接加在后面。如果当前元素`x`小于最后一个元素，那么在单调数组中找到大于`x`的第一个元素，将其更新为`x`

注意：最后单调数组中的元素不一定是最终的结果，但是其长度是最长的`LIS`长度。
如：`1 2 6 8 5`， 最后单调数组为：`1 2 5 8`， 但结果应该是`1 2 6 8`。

因此单调数组并不是为了求出最终的方案，而是看`LIS`最长能延伸到什么地方，之所以不断找到大于等于`x`的数然后替换为`x`，就是为今后的延伸留出更多的空间。

**时间复杂度$O(nlogn)$**
```c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        vector<int>a;
        for (auto x : nums) {
            if(a.empty() || x > a.back()) a.push_back(x);
            else    *lower_bound(a.begin(), a.end(), x) = x;
        }
        return a.size();
    }
};
```
---

### 2407. 最长递增子序列 II
在普通的`LIS`的基础[严格递增]上加了条件：
子序列中相邻元素的差值不超过 `k` 。
`nums = [4,2,1,4,3,4,5,8,15], k = 3`
`ans = 5, details = [1,3,4,5,8]`

核心思想：如果当前考虑的数是`a[i]`, 需要快速找到`dp[a[i] - k : a[i] - 1]`中的最大值`mx`, `dp[a[i]] = mx + 1`. 核心就是用线段树维护`dp`数组

#### 线段树快速求区间最大值
```c++
class Solution {
public:
    // 线段树求最大值模板
    static const int N = 1e5 + 10;
    using ll = long long;
    int m, p;

    // 线段树节点
    struct node{
        int l, r;
        int v; // 维护最大值
    };
    node tr[4 * N]; // 开4倍序列的大小
    void build(int u, int l, int r) {
        tr[u] = {l, r}; 
        if(l == r) return;
        int mid = l + r >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
    }

    void pushup(int u) { // 根据u的子节点信息更新u节点
        tr[u].v = max(tr[u << 1].v, tr[u << 1 | 1].v);
    }

    void modify(int u, int x, int v) {
        if(tr[u].l == tr[u].r) tr[u].v = v; // 叶子节点
        else {
            int mid = tr[u].l + tr[u].r >> 1;
            if(x <= mid) modify(u << 1, x, v); // 索引x在左子树
            else modify(u << 1 | 1, x, v); // 索引x在右子树
            pushup(u);
        }
    }
    int query(int u, int l, int r) {
        if(l <= tr[u].l && r >= tr[u].r) return tr[u].v; // 完全包含
        int mid = tr[u].l + tr[u].r >> 1;
        int val = 0;
        if(l <= mid) val = max(val, query(u << 1, l, r)); // 左节点与[l,r]有重叠部分，访问左节点
        if(r > mid) val = max(val, query(u << 1 | 1, l, r)); // 右节点与[l,r]有重叠部分，访问右节点
        return val;
    }

    int lengthOfLIS(vector<int>& nums, int k) {
        int mx = *max_element(nums.begin(), nums.end()); // 确认最多有多少个点
        build(1, 1, mx);
        for(auto x : nums) {
            int pre = x - k;
            int mx = query(1, pre, x - 1); // 找之前的最大值
            modify(1, x, mx + 1); // 状态转移
        }
        return query(1, 1, mx); // 所有dp[i]中的最大值
    }
};
```
---

### 673. 最长递增子序列的个数
`[1,3,5,4,7], ans = 有两个最长递增子序列，分别是 [1, 3, 4, 7] 和[1, 3, 5, 7]。`

```c++
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int>dp(n, 1), cnt(n, 1);
        int maxn = 1;
        for(int i = 1; i < n; i++) {
            for(int j = 0; j < i; j++) {
                if (nums[j] >= nums[i]) continue;
                if (dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1;
                    cnt[i] = cnt[j]; // 重置计数
                }
                else if(dp[j] + 1 == dp[i]) {
                    cnt[i] += cnt[j]; // 相当于有另一条路径得到最优解，加上
                }
            }
            maxn = max(maxn, dp[i]);
        }
        int res = 0;
        for(int i = 0; i < n; i++) {
            if(dp[i] == maxn) res += cnt[i]; // 把所有最长子序列的数量加起来
        }
        return res;
    }
};
```
---


### 674. 最长连续递增序列
题意：不仅要求递增，还要求连续。
返回最长的长度

#### 思路1：贪心
**时间复杂度$O(n)$**
```c++
class Solution {
public:
    int findLengthOfLCIS(vector<int>& nums) {
        int res = 1, ans = 1;
        for(int i = 1; i < nums.size(); i++) {
            if (nums[i] > nums[i - 1]) {
                res++;
                ans = max(ans, res);
            }
            else  res = 1;
        }
        return ans;
    }   
};
```

#### 思路2：DP
```c++
class Solution {
public:
    int findLengthOfLCIS(vector<int>& nums) {
        int n = nums.size();
        vector<int>dp(n, 1);
        int ans = 1;
        for (int i = 1; i < n; i++) {
            if (nums[i] > nums[i - 1]) dp[i] = dp[i - 1] + 1;
            ans = max(ans, dp[i]);
        }
        return ans;
    }
};
```
---

### 354. 俄罗斯套娃信封问题
`envelopes = [[5,4],[6,4],[6,7],[2,3]] ans = 3`
套娃是一个长方形，小的可以放在大的里面（长和宽均小于）
问最多套多少个。
很直接的思路就是先对第一维排序，然后对第二维求LIS

注意：因为第一维排序后并不能保证第一维严格单调，为了避免进行第二维选择LIS时，第一维的数值相同。在第一维相同时，第二维从大到小排序，从而保证不会选到第一维相同的。

```c++
class Solution {
public:
    int maxEnvelopes(vector<vector<int>>& v) {
        sort(v.begin(), v.end(), [&](vector<int>&a, vector<int>&b){
            if(a[0] == b[0]) return a[1] > b[1]; // 重点在于这里的排序规则
            return a[0] < b[0];
        });
        // LIS
        int n = v.size();
        vector<int>f;
        for(int i = 0; i < n; i++) {
            int x = v[i][1];
            if(f.empty() || x > f.back()) f.push_back(x);
            else *lower_bound(f.begin(), f.end(), x) = x;
        }
        return f.size();
    }
};
```

### 面试题 08.13. 堆箱子
把上述问题推广为3维的，长宽高都必须小于才能发放到上面，问堆的最大高度。
`box = [[1, 1, 1], [2, 2, 2], [3, 3, 3]], ans = 6`

```c++
class Solution {
public:
    int pileBox(vector<vector<int>>& box) {
        int n = box.size();
        sort(box.begin(), box.end());
        vector<int>f(n, 0);
        for(int i = 0; i < n; i++) f[i] = box[i][2];

        for(int i = 1; i < n; i++) {
            for(int j = 0; j < i; j++) {
                if(box[i][0] > box[j][0] && box[i][1] > box[j][1] && box[i][2] > box[j][2]) 
                f[i] = max(f[i], f[j] + box[i][2]);
            }
        }
        return *max_element(f.begin(), f.end());
    }
};
```