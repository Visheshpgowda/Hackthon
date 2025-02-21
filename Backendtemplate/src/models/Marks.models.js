import mongoose, { Schema } from "mongoose";

const marksSchema = new Schema({ 
    roll_no: {
        type: Number,
        required: true
    },
    Subject_name: {
        type: Schema.Types.ObjectId,
        ref: 'Subject',
        required: true
    },
    Exam_type: {
        type: String,
        enum: ['Midterm', 'Final'],
        required: true
    },
    marks: {
        type: Number,
        required: true
    },class: {
        type: String,
            required: true,
    },
    teacher: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    }
}, {
    timestamps: true
});


export const Marks = mongoose.model('Marks', marksSchema);