<font face= "楷体" size = 3>
---
<center><font face="楷体" size=6, color='red'> 树形DP </font> </center>

### 543. 二叉树的直径
直径的定义：任意两个结点路径长度中的最大值
基本思想：枚举从 `x` 往左儿子走的最长链和往右儿子走的最长链，这两条链可能会组成直径。枚举所有点就能找到答案。

```c++
class Solution {
public:
    int ans = 0;
    int maxdepth(TreeNode *root) {
        if(!root) return 0;
        int l = maxdepth(root->left);
        int r = maxdepth(root->right);
        ans = max(ans, l + r + 1);
        return max(l, r) + 1;
    }
    int diameterOfBinaryTree(TreeNode* root) {
        maxdepth(root);
        return ans - 1; // 路径 = 节点个数 - 1
    }
};
```
---

### 124. 二叉树中的最大路径和
找到二叉树中的最大的路径的和，节点值可能为负数
```c++
class Solution {
public:
    int maxPathSum(TreeNode* root) {
        unordered_map<TreeNode*, int>f;
        int ans = INT_MIN;
        function<int(TreeNode*)>DFS = [&](TreeNode* root){
            if(!root)  return 0;
            int lmax = max(DFS(root->left), 0); // 因为存在负数，所以可以不选左子树
            int rmax = max(DFS(root->right), 0);
            ans = max(ans, lmax + rmax + root->val); 
            f[root] = max(max(lmax, rmax) + root->val, root->val); // 是否不考虑当前节点往下的点
            return f[root];
        };    
        DFS(root);
        return ans;
    }
};
```

#### 学习树的直径
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

### 337. 打家劫舍 III
题意：一棵二叉树，树上的每个点都有对应的权值，每个点有两种状态（选中和不选中），问在不能同时选中有父子关系的点的情况下，能选中的点的最大权值和是多少。

#### 树形`DP`
通常形式为：`f[root]`

思路：
`f[root]`表示选`root`点的最大值， `g[root]`表示不选`root`点的最大值
因此状态转移：
```c++
f[root] = root->val +  g[root->left] + g[root->right];
g[root] = max(g[root->left], f[root->left]) + max(g[root->right],f[root->right]);
```
```c++
class Solution {
public:
    unordered_map<TreeNode*, int>f, g; // f[root]表示选root点的最大值， g[root]表示不选root点的最大值
    int rob(TreeNode* root) {
        DFS(root);
        return max(g[root], f[root]);
    }
    void DFS(TreeNode* root) { // 后序遍历
        if (!root) return;
        DFS(root->left);
        DFS(root->right);
        f[root] = root->val +  g[root->left] + g[root->right];
        g[root] = max(g[root->left], f[root->left]) + max(g[root->right], f[root->right]);
    }
};
```
时间复杂度$O(n)$:相当于对二叉树做了一次后序遍历
空间复杂度$O(n)$：栈空间 + 哈希表空间

**另一种写法：**  (考虑到每个节点在计算时，仅仅依赖于他的子节点)
```c++
class Solution {
public:
    vector<int> DFS(TreeNode* root) { // 0:不选，1:选
        if (!root) return {0, 0};
        vector<int> left = DFS(root->left);
        vector<int> right = DFS(root->right);
        int res1 = max(left[0], left[1]) + max(right[0], right[1]);
        int res2 = root->val + left[0] + right[0];
        return {res1, res2};
    }
    int rob(TreeNode* root) {
        auto ans = DFS(root);
        return max(ans[0], ans[1]);
    }
};
```
```c++
class Solution {
public:
    using pii = pair<int, int>;
    pii DFS(TreeNode* root) {
        if(!root) return {0, 0};
        auto [lf, lg] = DFS(root->left);
        auto [rf, rg] = DFS(root->right);

        int f = lg + rg + root->val;
        int g = max(lg, lf) + max(rg, rf);
        return {f, g};
    }
    int rob(TreeNode* root) {
        auto [f, g] = DFS(root);
        return max(f, g);
    }
};
```
时间复杂度$O(n)$; 空间复杂度$O(n)$:虽说省去了哈希表的空间，但是还有栈空间

---

### acwing 285. 没有上司的舞会
给定一颗树，每个点有一个价值。
问根节点和它的儿子不能同时选的最大价值。
即打家劫舍III 从二叉树变为了多叉树

```c++
#include<bits/stdc++.h>
using namespace std;
const int N = 6010;
int happy[N];
bool isson[N];

int h[N], ne[N], e[N], idx;
void add(int a, int b) {
    e[idx] = b, ne[idx] = h[a], h[a] = idx++;
}

int f[N], g[N]; //分别表示选根节点和不选根节点的最大价值

void DFS(int u) {
    f[u] = happy[u];
    for(int i = h[u]; i != -1; i = ne[i]) {
        int j = e[i];
        
        DFS(j);
        f[u] += g[j];
        g[u] += max(f[j], g[j]);
    }
}
int main() {
    int n;
    cin >> n;
    memset(h, -1, sizeof h);
    for(int i = 1; i <= n; i++) cin >> happy[i]; // 读入价值
    for(int i = 1; i < n; i++) {
        int a, b;
        cin >> a >> b;
        add(b, a);
        isson[a] = 1;  // 便于找root
    }
    int root = -1;
    for(int i = 1; i <= n; i++) if (!isson[i]) root = i; // 找root
    DFS(root);
    cout << max(f[root], g[root]) << endl;
    return 0;
}
```
---

