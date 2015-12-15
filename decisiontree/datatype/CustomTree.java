package hw1.decisiontree.datatype;

public class CustomTree {

	private Integer id;
	private Double threshold;
	private Integer featop;
	private String type;
	private CustomTree right,left;
	private Double label;
	public CustomTree(int nodeno,double thresh,int featno,double l)
	{
		id=nodeno;
		threshold=thresh;
		featop=featno;
		type="end";
		label=l;
	}
	public CustomTree(CustomTree n) {
		id=n.getNodeno();
		threshold=n.getThreshold();
		featop=n.getFeatno();
		label=n.getLabel();
		type=n.getType();
		left=null;
		right=null;
	}
	public CustomTree() {
		type="end";
		left=null;
		right=null;
	}
	public double getThreshold(){
		return threshold;
	}
	public int getFeatno(){
		return featop;
	}
	public int getNodeno(){
		return id;
	}
	public String getType()
	{
		return type;
	}
	public void getNode(CustomTree node)
	{
		id=node.getNodeno();
		threshold=node.getThreshold();
		featop=node.getFeatno();
		label=node.getLabel();
		type=node.getType();
		left=null;
		right=null;
	}
	public Double getLabel()
	{
		return label;
	}
	public CustomTree getRight(){
		return right;
	}
	public CustomTree getLeft(){
		return left;
	}
	public void setType(){
		type="tree";
	}
	public void setRight(CustomTree n){
		setType();
		this.right=new CustomTree(n);
	}
	public void setLeft(CustomTree n){
		setType();
		this.left=new CustomTree(n);
	}
	public void placenode(int nodeno,CustomTree pnode)
	{
       int temp=nodeno;
       CustomTree tn=this;
		while(true){
			//System.out.println(tn.getNodeno()+"   "+nodeno);
			if(tn.getNodeno()==nodeno)
				{
				 tn.getNode(pnode);
				 return;
				}
			if(tn.getNodeno()*2==nodeno)
				{
				 tn.setLeft(pnode);
				 return;
				}
			if(tn.getNodeno()*2+1==nodeno)
				{
				 tn.setRight(pnode);
				 return;
				}
			temp=nodeno;
			while(true)
			{
				if(temp/2==tn.getNodeno())
					break;
				temp/=2;
			}
			//System.out.println(temp+"h");
			
			if(temp%2==1)
			{
			 tn=tn.getRight();	
			}
			else
			{
			 tn=tn.getLeft();	
			}
		}
        //tn.getNode(pnode);
	}
}
