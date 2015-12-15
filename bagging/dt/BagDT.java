package hw4.bagging.dt;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import hw1.decisiontree.datatype.CustomTree;

public class BagDT {

	static int len=4601;
	static int featlen=57;
	static int classno=2;
	static int trainlen,testlen;
	static double[][] features = new double[len][featlen];
	static double[] labels = new double[len];
	static ArrayList<double[][]> featklist = new ArrayList<double[][]>();
	static ArrayList<double[]> labelsklist = new ArrayList<double[]>();
	static ArrayList<Integer> indexlist = new ArrayList<Integer>();
	static double[][] featurestrain = new double[1][featlen];
	static double [] labelstrain = new double[1];
	static double [][] featurestest= new double[1][featlen];
	static double [] labelstest= new double[1];
	static int bags=50;
	static int samplingpercent=100;
	static int bagtrainlen;
	static ArrayList<ArrayList<ArrayList<Double>>> trainbags= new ArrayList<ArrayList<ArrayList<Double>>>();
	static ArrayList<ArrayList<Double>> labelbags= new ArrayList<ArrayList<Double>>();
	static ArrayList<ArrayList<Integer>> indexbags= new ArrayList<ArrayList<Integer>>();
	static ArrayList<ArrayList<Double>> test= new ArrayList<ArrayList<Double>>();
	static ArrayList<Double> testlabel= new ArrayList<Double>();
	static ArrayList<CustomTree> bagtree= new ArrayList<CustomTree>();
	private static void readData()
	{
		int one=0,zero=0;
		try {
			FileReader featureread = new FileReader("data/spambase.data");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				indexlist.add(ind);
				feats=sCurrentLine.split(",");
				for(int i=0;i<feats.length-1;i++)
				{
					features[ind][i]=Double.parseDouble(feats[i]);

				}
				labels[ind]=Double.parseDouble(feats[feats.length-1]);
				if(labels[ind]==0.0)
					zero++;
				else
					one++;
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		//System.out.println(zero+"   "+one+"  "+(double) zero/(zero+one));
	}
	private static void randSplits(int folds)
	{
		HashMap<Integer, Boolean> seen = new HashMap<Integer, Boolean>();
		int kfold=len/folds,listno=0,one=0,zero=0;
		Random n = new Random(); 
		while(indexlist.size()!=0){
			one=0;zero=0;
			int r=Math.min(kfold, indexlist.size());
			double[][] featurestemp = new double[r][featlen];
			double [] labelstemp = new double[r];
			for (int i = 0; i < kfold; i++) {
				if(indexlist.size()==0)
					break;
				int ind=n.nextInt(indexlist.size());
				int tmpind=indexlist.get(ind);
				if(seen.containsKey(tmpind))
					System.out.println("Error: bad Coder");
				else
					seen.put(tmpind, true);
				for (int j = 0; j < featlen; j++) {
					featurestemp[i][j]=features[tmpind][j];
				}
				labelstemp[i]=labels[tmpind];
				if(labels[tmpind]==0.0)
					zero++;
				else if(labels[tmpind]==1.0)
					one++;
				else
					System.out.println("Error: bad Coder");

				indexlist.remove(ind);
			}
			//System.out.println(zero+"   "+one+"   "+(double) zero/(zero+one));
			featklist.add(listno,featurestemp);
			labelsklist.add(listno,labelstemp);
			listno++;
		}
	}
	private static void uniformSplits(int folds){
		ArrayList<ArrayList<double[]>> tft= new ArrayList<ArrayList<double[]>>();
		ArrayList<ArrayList<Double>> tlt= new ArrayList<ArrayList<Double>>();
		for (int i = 0; i < folds; i++) {
			tft.add(new ArrayList<double[]>());
			tlt.add(new ArrayList<Double>());
		}
		for (int i = 0; i < len; i++) {
			ArrayList<double[]> temp = tft.get(i%folds);
			temp.add(features[i]);
			tft.set(i%folds, temp);
			ArrayList<Double> lemp = tlt.get(i%folds);
			lemp.add(labels[i]);
			tlt.set(i%folds,lemp);
		}
		if(tft.size()!=tlt.size())
			System.out.println("Error: BadCode");
		int one=0,zero=0,listno=0;
		for (int i = 0; i < tft.size(); i++) {
			one=0;zero=0;
			double[][] featurestemp = new double[tft.get(i).size()][featlen];
			double [] labelstemp = new double[tlt.get(i).size()];
			for (int j = 0; j < tft.get(i).size(); j++) {
				featurestemp[j]=tft.get(i).get(j);
				labelstemp[j]=tlt.get(i).get(j);
				if(labelstemp[j]==0)
					zero++;
				else
					one++;
			}
			//System.out.println(zero+"   "+one+"   "+(double) zero/(zero+one));
			featklist.add(listno,featurestemp);
			labelsklist.add(listno,labelstemp);
			listno++;
		}
	}
	private static void pickKfold(int k){
		k--;
		testlen=labelsklist.get(k).length;
		trainlen=len-testlen;
		featurestrain=new double[trainlen][featlen];
		labelstrain=new double[trainlen];
		featurestest=new double[testlen][featlen];
		labelstest=new double[testlen];
		int ind=0;
		for (int i = 0; i < featklist.size(); i++) {
			if(i==k)
			{
				featurestest=featklist.get(k);
				labelstest=labelsklist.get(k);
				continue;
			}
			double[][] tf = featklist.get(i);
			double[] tl = labelsklist.get(i);
			for (int j = 0; j < tl.length; j++) {
				featurestrain[ind]=tf[j];
				labelstrain[ind]=tl[j];
				ind++;
			}
		}
	}
	private static void setBags(){
		bagtrainlen=(trainlen*samplingpercent)/100;
		ArrayList<ArrayList<Double>> temptrain = new ArrayList<ArrayList<Double>>();
		for (int i = 0; i < trainlen; i++) {
			ArrayList<Double> onedatapoint=new ArrayList<Double>();
			for (int j = 0; j < featlen; j++) {
				onedatapoint.add(j, featurestrain[i][j]);
			}
			temptrain.add(i, onedatapoint);
		}
		Random n = new Random();
		for (int i = 0; i < bags; i++) {
			int ind=0;
			ArrayList<ArrayList<Double>> bag = new ArrayList<ArrayList<Double>>();
			ArrayList<Double> baglabel=new ArrayList<Double>();
			ArrayList<Integer> localindex=new ArrayList<Integer>();
			while(ind<bagtrainlen)
			{
				int step=n.nextInt(trainlen);
				bag.add(ind,temptrain.get(step));
				baglabel.add(ind, labelstrain[step]);
				localindex.add(ind);
				ind++;
			}
			ArrayList<ArrayList<Double>> inversebag = new ArrayList<ArrayList<Double>>();
			for (int j = 0; j < featlen; j++) {
				ArrayList<Double> onedata = new ArrayList<Double>();
				for (int k = 0; k < bagtrainlen; k++) {
					onedata.add(k, bag.get(k).get(j));
				}
				inversebag.add(j,onedata);
			}
			trainbags.add(i, inversebag);
			labelbags.add(i,baglabel);
			indexbags.add(i, localindex);
		}
		for (int i = 0; i < featlen; i++) {
			ArrayList<Double> onedatapoint=new ArrayList<Double>();
			for (int j = 0; j < testlen; j++) {
				onedatapoint.add(j, featurestest[j][i]);
			}
			test.add(i, onedatapoint);
		}
		for (int j = 0; j < testlen; j++) {
			testlabel.add(j, labelstest[j]);
		}
	}
	private static Double CalculateEntropy(HashMap<Double, Integer> count,int total)
	{
		double prob=0.0,entropy=0.0;
		for (Double key:count.keySet()) {
			prob=count.get(key)/(double) total;
			if(prob!=0)
				entropy+=(-1)*prob*(Math.log10(prob)/Math.log10(2));
		}
		return entropy;
	}
	private static Double InformationGain(Double entropy,HashMap<Double, Integer> less,HashMap<Double, Integer> notless,int l,int nl,int total){
		return entropy-(double)l/(double)total*CalculateEntropy(less, l)-(double)nl/(double)total*CalculateEntropy(notless, nl);
	}
	private static Double[] FindThreshold(ArrayList<Double> labels,ArrayList<Double> feat){
		int total=labels.size(),l=0,nl=total;
		Map<Double, HashMap<Double,Integer>> featlabel = new HashMap<Double,HashMap<Double,Integer>>();
		int eachval=0;
		Double featkey=0.0;
		for(int i=0;i<total;i++)
		{
			HashMap<Double,Integer> tempval = featlabel.get(feat.get(i));
			featkey=labels.get(i);
			if(tempval==null)
			{
				tempval=new HashMap<Double,Integer>();
				tempval.put(featkey, 1);
			}
			else
			{
				eachval=0;
				if(tempval.containsKey(featkey))
				{
					eachval=tempval.get(featkey);
				}
				tempval.put(featkey, eachval+1);
			}
			featlabel.put(feat.get(i),tempval);
		}
		featlabel=new TreeMap<Double,HashMap<Double,Integer>>(featlabel);
		HashMap<Double, Integer> count=new HashMap<Double, Integer>();
		double temp=0.0;
		for (int i = 0; i < total; i++) {
			temp=labels.get(i);
			if(count.containsKey(temp))
			{
				count.put(temp,(count.get(temp)+1));
			}
			else
			{
				count.put(temp,1);
			}
		}
		temp=CalculateEntropy(count, total);
		double ig=0.0,curthreshold=0.0,threshold=0.0,temp1=0.0,temp2=0.0,tempig=0.0;
		HashMap<Double, Integer> less=new HashMap<Double, Integer>();
		for(Double key:count.keySet())
		{
			less.put(key, 0);
		}
		int i=0;
		//System.out.println(featlabel.size());
		for (Double key:featlabel.keySet()) {
			HashMap<Double,Integer> tempval = featlabel.get(key);
			if(i==0)
			{
				for (Double tempvalkey:tempval.keySet()) 
				{
					less.put(tempvalkey,less.get(tempvalkey)+tempval.get(tempvalkey));
					count.put(tempvalkey,(count.get(tempvalkey)-tempval.get(tempvalkey)));
					l+=tempval.get(tempvalkey);nl-=tempval.get(tempvalkey);
					//System.out.println("here  "+key+"   "+tempvalkey+"   "+less.get(tempvalkey)+"  "+count.get(tempvalkey)+"  "+tempval.get(tempvalkey));
				}
				temp1=key;
				i++;
				continue;
			}
			temp2=key;
			curthreshold=(temp1+temp2)/2.0;
			tempig=InformationGain(temp, less, count, l, nl, total);
			//System.out.println(curthreshold+"   "+l+"  "+nl+"  "+total+"  "+tempig);
			if(tempig>ig)
			{
				ig=tempig;
				threshold=curthreshold;
			}
			temp1=temp2;
			for (Double tempvalkey:tempval.keySet()) 
			{
				less.put(tempvalkey,less.get(tempvalkey)+tempval.get(tempvalkey));
				count.put(tempvalkey,(count.get(tempvalkey)-tempval.get(tempvalkey)));
				l+=tempval.get(tempvalkey);nl-=tempval.get(tempvalkey);
				//System.out.println("here  "+key+"   "+tempvalkey+"   "+less.get(tempvalkey)+"  "+count.get(tempvalkey)+"  "+tempval.get(tempvalkey));
			}

		}
		Double[] all={threshold,ig};
		return all;
	}
	private static Double[] FindBestSplit(ArrayList<Double> labels,ArrayList<ArrayList<Double>> feats){
		double ig=0.0,curfeat=0.0,tempig=0.0,thresh=0.0,tempthresh=0.0;
		Double[] result={0.0,0.0};
		for (int i = 0; i < feats.size(); i++) {
			//System.out.println(i);
			result=FindThreshold(labels,feats.get(i));
			tempig=result[1];
			tempthresh=result[0];
			//System.out.println(tempig+"   "+tempthresh);
			if(tempig>ig)
			{
				ig=tempig;
				curfeat=i;
				thresh=tempthresh;
			}
		}
		Double[] all={curfeat,thresh,ig};
		return all;
	}
	private static Double[] CreateFeatMat(ArrayList<Double> labels,ArrayList<Integer> index,ArrayList<ArrayList<Double>> feats){
		ArrayList<ArrayList<Double>> featuresnew = new ArrayList<ArrayList<Double>>();
		ArrayList<Double> labelsnew = new ArrayList<Double>();
		for(int i=0;i<index.size();i++)
		{
			labelsnew.add(i,labels.get(index.get(i)));
		}
		for (int i = 0; i < feats.size(); i++) {
			ArrayList<Double> temp = new ArrayList<Double>();
			for(int j=0;j<index.size();j++)
			{
				//System.out.println(i+" "+index.get(j)+" "+feats.size());
				temp.add(j,feats.get(i).get(index.get(j)));
			}
			featuresnew.add(i,temp);
		}
		return FindBestSplit(labelsnew, featuresnew);
	}
	private static ArrayList<ArrayList<Integer>> SplitOnThreshold(Double threshold,ArrayList<Double> feat,ArrayList<Integer> index){
		ArrayList<Integer> lindex = new ArrayList<Integer>();
		ArrayList<Integer> rindex = new ArrayList<Integer>();
		int lind=0,rind=0;
		for (int i = 0; i < index.size(); i++) {
			//System.out.println(i);
			if(feat.get(index.get(i))<threshold)
			{
				//System.out.println(index.get(i));
				lindex.add(lind,index.get(i));
				lind++;
			}
			else
			{
				rindex.add(rind,index.get(i));
				rind++;	
			}
		}
		ArrayList<ArrayList<Integer>> all = new ArrayList<ArrayList<Integer>>(); 
		all.add(0,lindex);
		all.add(1,rindex);
		return all;
	}
	private static Double MaxFeature(ArrayList<Double> labels,ArrayList<Integer> index){
		int max=0;double ans=0.0;
		HashMap<Double, Integer> maxc = new HashMap<Double, Integer>();
		for (int i = 0; i < index.size(); i++) {
			if(maxc.containsKey(labels.get(index.get(i))))
			{
				maxc.put(labels.get(index.get(i)), maxc.get(labels.get(index.get(i)))+1);
			}
			else
				maxc.put(labels.get(index.get(i)), 1);
		}
		//System.out.println("size "+maxc.size()+"   "+index.size());
		for(Double key:maxc.keySet())
		{
			//System.out.println(key+"    "+maxc.get(key));
			if(maxc.get(key)>max)
			{
				max=maxc.get(key);
				ans=key;
			}
		}
		return ans;
	}
	private static CustomTree BuildNode(ArrayList<ArrayList<Double>> features,ArrayList<Double> labels,ArrayList<Integer> index,int nodeno){
		Double[] all= CreateFeatMat(labels, index, features);
		CustomTree node = new CustomTree(nodeno,all[1],all[0].intValue(),MaxFeature(labels, index));
		//System.out.println(nodeno+"   "+node.getLabel()+" "+node.getType()+"   "+node.getThreshold());
		return node;
	}
	private static CustomTree BuildTree(ArrayList<ArrayList<Double>> features,ArrayList<Double> labels,HashMap<Integer, ArrayList<Integer>> indexlist){
		ArrayList<Integer> nodelist = new ArrayList<Integer>();
		int nodeno=1;
		nodelist.add(nodeno);
		CustomTree tree = new CustomTree();
		for (int i = 0; i < 103; i++) {
			nodeno=nodelist.get(i);
			CustomTree node = new CustomTree();
			node=BuildNode(features, labels, indexlist.get(nodeno), nodeno);
			//System.out.println(i+" a  here "+node.getNodeno());
			if(i==0)
				tree.getNode(node);
			else
				tree.placenode(nodeno, node);
			//.out.println(i+" b  here "+tree.getNodeno());
			ArrayList<ArrayList<Integer>> ans = SplitOnThreshold(node.getThreshold(), features.get(node.getFeatno()), indexlist.get(nodeno));
			//.out.println(i+" c  here ");
			//System.out.println(node.getFeatno());
			//System.out.println(i+"   "+ans.get(0).size()+"   "+ans.get(1).size());
			indexlist.put(nodeno*2, ans.get(0));
			indexlist.put(nodeno*2+1, ans.get(1));
			nodelist.add(nodeno*2);
			nodelist.add(nodeno*2+1);
		}
		return tree;
	}
	private static Double[] AccuracyPredict(ArrayList<ArrayList<Double>> featurestest,ArrayList<Double> labelstest,CustomTree tree){
		int total=labelstest.size();
		int wrong=0;
		double mse=0.0;
		for (int i = 0; i < total; i++){
			double thresh=0.0,predict=0.0;int featno=0;
			CustomTree ttemp = new CustomTree();
			ttemp=tree;
			while(true)
			{
				featno=ttemp.getFeatno();
				thresh=ttemp.getThreshold();
				//System.out.println(ttemp.getNodeno()+"   "+featno+"   "+thresh+"   "+featurestest.get(featno).get(i)+"   "+ttemp.getType()+"   "+ttemp.getLabel());
				if(ttemp.getType().contentEquals("tree"))
				{
					if(featurestest.get(featno).get(i)<thresh)
					{
						ttemp=ttemp.getLeft();
					}
					else
					{
						ttemp=ttemp.getRight();
					}
				}
				else
				{
					predict=ttemp.getLabel();
					break;
				}
			}
			if(labelstest.get(i)!=predict)
			{
				wrong++;
				mse+=(predict-labelstest.get(i))*(predict-labelstest.get(i));
			}
		}
		Double all[]={mse/(double) total,(100-wrong*100/(double) total),(double) wrong};
		return all;
	}
	private static Double[] AccuracyBagPredict(ArrayList<ArrayList<Double>> featurestest,ArrayList<Double> labelstest){
		int total=labelstest.size();
		int wrong=0;
		double mse=0.0;
		for (int i = 0; i < total; i++){
			double thresh=0.0,predict=0.0;int featno=0;
			HashMap<Double, Integer> lc = new HashMap<Double, Integer>();
			for (int j = 0; j < bags; j++) {
				CustomTree ttemp = new CustomTree();
				ttemp=bagtree.get(j);
				while(true)
				{
					featno=ttemp.getFeatno();
					thresh=ttemp.getThreshold();
					//System.out.println(ttemp.getNodeno()+"   "+featno+"   "+thresh+"   "+featurestest.get(featno).get(i)+"   "+ttemp.getType()+"   "+ttemp.getLabel());
					if(ttemp.getType().contentEquals("tree"))
					{
						if(featurestest.get(featno).get(i)<thresh)
						{
							ttemp=ttemp.getLeft();
						}
						else
						{
							ttemp=ttemp.getRight();
						}
					}
					else
					{
						predict=ttemp.getLabel();
						break;
					}
				}
				if(lc.containsKey(predict))
				{
					lc.put(predict, lc.get(predict)+1);
				}
				else
				{
					lc.put(predict, 1);
				}
			}
			int max=0;
            for (Double key:lc.keySet()) {
				if(lc.get(key)>max)
				{
					max=lc.get(key);
					predict=key;
				}
			}
			if(labelstest.get(i)!=predict)
			{
				wrong++;
				mse+=(predict-labelstest.get(i))*(predict-labelstest.get(i));
			}
		}
		Double all[]={mse/(double) total,(100-wrong*100/(double) total),(double) wrong};
		return all;
	}

	public static void main(String[] args) {
		readData();
		uniformSplits(10);
		pickKfold(1);
		setBags();
		for (int i = 0; i < bags; i++) {
			HashMap<Integer, ArrayList<Integer>> indexlist = new HashMap<Integer, ArrayList<Integer>>();
			indexlist.put(1, indexbags.get(i));
			//System.out.println(trainbags.get(i).get(0).size()+"  "+indexbags.get(i).size()+"  "+labelbags.get(i).size());
			bagtree.add(BuildTree(trainbags.get(i), labelbags.get(i), indexlist));
			System.out.println(i);
		}
		double sum=0.0,msete=0.0;
		for (int i = 0; i < bags; i++) {
			Double allte[]=AccuracyPredict(test, testlabel, bagtree.get(i));
			sum+=allte[1];
			msete+=allte[0];
			//System.out.println(allte);
		}
		sum/=(double) bags;
		msete/=(double) bags;
		System.out.println("Accuracy: "+sum);
		System.out.println("Error: "+msete);
		Double allte[]=AccuracyBagPredict(test, testlabel);
		System.out.println("Accuracy: "+allte[1]);
		System.out.println("Error: "+allte[0]);
		
	}

}
