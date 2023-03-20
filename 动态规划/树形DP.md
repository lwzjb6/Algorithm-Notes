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
