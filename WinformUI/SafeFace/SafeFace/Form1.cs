using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System;
using System.Windows.Forms;
using Sunny.UI;
using System.IO;
using System.Net.Sockets;
using System.Net;

namespace SafeFace
{

    public partial class Form1 : UIHeaderAsideMainFrame
    {
        public Form1()
        {
            InitializeComponent();
            int pageIndex = 1000;
            // Aside创建Controls节点，并给对应的page标号为pageIndex
            // FMain1需要继承UIPage或者 UITitlePage
            // 将Aside和指定的Main对应关联
            //TreeNode treeNode = new TreeNode();
            UIPage page = AddPage(new Fmain1(), pageIndex);
            TreeNode treeNode = Aside.CreateNode("Fmain1", pageIndex);
            treeNode.Text = "人脸识别";
            // 需要将修改后的TreeNode重新装入
            //Aside.SetNodeItem(treeNode, new NavMenuItem(page));

            pageIndex = 2000;
            // Aside创建Controls节点，并给对应的page标号为pageIndex
            page = AddPage(new Fmain2(), pageIndex);
            treeNode = Aside.CreateNode("Fmain2", pageIndex);
            treeNode.Text = "活体检测";
            //Aside.SetNodeItem(treeNode, new NavMenuItem(page));

            pageIndex = 3000;
            // Aside创建Controls节点，并给对应的page标号为pageIndex
            page = AddPage(new Fmain3(), pageIndex);
            treeNode = Aside.CreateNode("Fmain3", pageIndex);
            treeNode.Text = "人像分割";
            //Aside.SetNodeItem(treeNode, new NavMenuItem(page));

    }
        
        private void Form1_Load(object sender, EventArgs e)
        {

        }

    }
}
