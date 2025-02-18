import mongoose,{Schema} from "mongoose";

const marksSchema = new Schema({
    Student_name:{
        type: Schema.Types.ObjectId,
        ref: 'Student',
        required: true
    },
    subject:{
        type:String,
        required:true
    },
    Exam_type:{
        type:String,
        enum:['Midterm','Final'],
        required:true
    },
    marks:{
        type:Number,
        required:true
    }
},{
    timestamps:true
});


export const marksModel=mongoose.model('Marks',marksSchema);
