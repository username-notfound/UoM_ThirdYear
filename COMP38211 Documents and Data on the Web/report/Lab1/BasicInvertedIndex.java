/**
 * Basic Inverted Index
 * 
 * This Map Reduce program should build an Inverted Index from a set of files.
 * Each token (the key) in a given file should reference the file it was found 
 * in. 
 * 
 * The output of the program should look like this:
 * sometoken [file001, file002, ... ]
 * 
 * @author Kristian Epps
 */
package uk.ac.man.cs.comp38211.exercise;

import java.io.*;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import uk.ac.man.cs.comp38211.io.array.ArrayListOfLongsWritable;
import uk.ac.man.cs.comp38211.io.array.ArrayListWritable;
import uk.ac.man.cs.comp38211.io.pair.PairOfIntString;
import uk.ac.man.cs.comp38211.io.pair.PairOfStrings;
import uk.ac.man.cs.comp38211.io.pair.PairOfWritables;
import uk.ac.man.cs.comp38211.ir.Stemmer;
import uk.ac.man.cs.comp38211.ir.StopAnalyser;
import uk.ac.man.cs.comp38211.util.XParser;

public class BasicInvertedIndex extends Configured implements Tool
{
    private static final Logger LOG = Logger
            .getLogger(BasicInvertedIndex.class);

    public static class Map extends 
            Mapper<Object, Text, Text, Text>
    {

        // INPUTFILE holds the name of the current file
        private final static Text INPUTFILE = new Text();
        
        // TOKEN should be set to the current token rather than creating a 
        // new Text object for each one
        @SuppressWarnings("unused")
        private final static Text TOKEN = new Text();

        // The StopAnalyser class helps remove stop words
        @SuppressWarnings("unused")
        private StopAnalyser stopAnalyser = new StopAnalyser();
        
        // The stem method wraps the functionality of the Stemmer
        // class, which trims extra characters from English words
        // Please refer to the Stemmer class for more comments
        @SuppressWarnings("unused")
        private String stem(String word)
        {
            Stemmer s = new Stemmer();

            // A char[] word is added to the stemmer with its length,
            // then stemmed
            s.add(word.toCharArray(), word.length());
            s.stem();

            // return the stemmed char[] word as a string
            return s.toString();
        }
        
        private String caseFolding(String token) {
        	// Condition1: a word with the first letter in upper case
        	// it is more likely that it is the first word of a sentence, which means we can turn it to lower case
        	// if the capital letter is in the middle, it may because of mistyping, we can still turn it to lower case
        	// Condition2: all the letter in this word is in upper case
        	// we probably should keep the form, such as BBC, DVD
        	int numOfCapitalLetter = 0;
        	char[] c = token.toCharArray();
        	for (int i = 0; i < token.length(); i++) {
      
        		if (Character.isUpperCase(c[i])) {numOfCapitalLetter++; }
        		if (numOfCapitalLetter == token.length()) 
        			return token;
        	}
        	return token.toLowerCase();
			
        	
        } // case folding
        
        // This method gets the name of the file the current Mapper is working
        // on
        @Override
        public void setup(Context context)
        {
            String inputFilePath = ((FileSplit) context.getInputSplit()).getPath().toString();
            String[] pathComponents = inputFilePath.split("/");
            INPUTFILE.set(pathComponents[pathComponents.length - 1]);
        }
         
        // TODO
        // This Mapper should read in a line, convert it to a set of tokens
        // and output each token with the name of the file it was found in
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException
        {
        	
            String line = value.toString();
            // remove all non-alphabetic characters
            //line = line.replaceAll("[^a-zA-Z.]", " "); 
            StringTokenizer stringTokenizer = new StringTokenizer(line);
            
            /* Positional indexing - record where the token lies in the document
            */
            IntWritable position = new IntWritable(0);
            HashMap<String, ArrayListWritable<IntWritable>> map = new HashMap<String, ArrayListWritable<IntWritable>>();
        	PairOfWritables<PairOfIntString, ArrayListWritable<IntWritable>> filenameAndPosition
                    = new PairOfWritables<PairOfIntString, ArrayListWritable<IntWritable>>();
            PairOfIntString countAndFilename = new PairOfIntString();
            int count = stringTokenizer.countTokens();

            //for(int position = 0; stringTokenizer.hasMoreTokens(); position++) {
            while (stringTokenizer.hasMoreTokens()) {
            	position.set(position.get() + 1);

            	String token = stringTokenizer.nextToken();
            	token = token.replaceAll("[^a-zA-Z]", "");
            	// case folding - should all terms be lower case, or do some need to remain upper case
            	// token = token.toLowerCase();
            	token = caseFolding(token);
            	
            	// deal with the stopword
            	// Reason why I do stopword removal before stemming is that
            	// for example, the token is 'was', it will become 'wa' after stemming
            	// so it will remain in my token list with no useful meaning
            	if(!stopAnalyser.isStopWord(token)) {  // if the token is a stop word, then it will break this iteration 
            		// stemming
            		token = stem(token);

                    /* Postional Indexing */
                    IntWritable posIndex = new IntWritable(position.get());
                    ArrayListWritable<IntWritable> posArray = new ArrayListWritable<IntWritable>();
                    if (map.containsKey(token)){
                        map.get(token).add(posIndex);
                    } // if
                    else{
                        posArray.add(posIndex);
	        			map.put(token, posArray);
                    } // else

            	
            		// The only one-letter words in English are a and I, which is not a good token
            		// So there I check the length of token to filter white space and a/I
            		if (token.length() > 2) {	
                        countAndFilename.set(count, INPUTFILE.toString());     
            			TOKEN.set(token);
                        filenameAndPosition.set(countAndFilename,map.get(token)); 
                        // context.write(TOKEN,filenameAndPosition); -- not working
            			context.write(TOKEN, INPUTFILE);
            		} // if
            	} // if is not a stopword
            } // while
            
        } //map method
        	
    } // map class

    public static class Reduce extends Reducer<Text, Text, Text, ArrayListWritable<Text>>
    {

        // TODO
        // This Reduce Job should take in a key and an iterable of file names
        // It should convert this iterable to a writable array list and output
        // it along with the key
        public void reduce(
                Text key,
                Iterable<Text> values,
                Context context) throws IOException, InterruptedException
        {
        	Iterator<Text> iterator = values.iterator();
        	ArrayListWritable<Text> array = new ArrayListWritable<Text>();

            /* Positioanl indexing 
            */
            int countOccurrences = 0;
 
            /* TFIDF
             * token i, document j
             * TF = number of occurrences of i in j / total number of tokens in j
             * IDF = log(total number of documents / number of documents containing i + 1)
            */
            float TF, IDF;
            int numOfDoc = 6;


        	// loop for each iterable of file names
        	while (iterator.hasNext()) {
                countOccurrences++;
        		Text t = iterator.next();
        		Text s = new Text(t.toString());
        		if (!array.contains(s)){
       
        			array.add(s);
        		} // if
        	} // while

        	context.write(key, array);

        }
    }

    // Lets create an object! :)
    public BasicInvertedIndex()
    {
    }

    // Variables to hold cmd line args
    private static final String INPUT = "input";
    private static final String OUTPUT = "output";
    private static final String NUM_REDUCERS = "numReducers";

    @SuppressWarnings({ "static-access" })
    public int run(String[] args) throws Exception
    {
        
        // Handle command line args
        Options options = new Options();
        options.addOption(OptionBuilder.withArgName("path").hasArg()
                .withDescription("input path").create(INPUT));
        options.addOption(OptionBuilder.withArgName("path").hasArg()
                .withDescription("output path").create(OUTPUT));
        options.addOption(OptionBuilder.withArgName("num").hasArg()
                .withDescription("number of reducers").create(NUM_REDUCERS));

        CommandLine cmdline = null;
        CommandLineParser parser = new XParser(true);

        try
        {
            cmdline = parser.parse(options, args);
        }
        catch (ParseException exp)
        {
            System.err.println("Error parsing command line: "
                    + exp.getMessage());
            System.err.println(cmdline);
            return -1;
        }

        // If we are missing the input or output flag, let the user know
        if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT))
        {
            System.out.println("args: " + Arrays.toString(args));
            HelpFormatter formatter = new HelpFormatter();
            formatter.setWidth(120);
            formatter.printHelp(this.getClass().getName(), options);
            ToolRunner.printGenericCommandUsage(System.out);
            return -1;
        }

        // Create a new Map Reduce Job
        Configuration conf = new Configuration();
        Job job = new Job(conf);
        String inputPath = cmdline.getOptionValue(INPUT);
        String outputPath = cmdline.getOptionValue(OUTPUT);
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer
                .parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        // Set the name of the Job and the class it is in
        job.setJobName("Basic Inverted Index");
        job.setJarByClass(BasicInvertedIndex.class);
        job.setNumReduceTasks(reduceTasks);
        
        // Set the Mapper and Reducer class (no need for combiner here)
        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);
        
        // Set the Output Classes
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(ArrayListWritable.class);

        // Set the input and output file paths
        FileInputFormat.setInputPaths(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));
        
        // Time the job whilst it is running
        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime)
                / 1000.0 + " seconds");

        // Returning 0 lets everyone know the job was successful
        return 0;
    }

    public static void main(String[] args) throws Exception
    {
        ToolRunner.run(new BasicInvertedIndex(), args);
    }
}
