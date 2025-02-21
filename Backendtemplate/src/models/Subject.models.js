import mongoose from 'mongoose';

const { Schema, model } = mongoose;

const subjectSchema = new Schema({
    Subject_name: {
        type: String,
        required: true
    },
    teacher: [{
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    }],
});

const Subject = model('Subject', subjectSchema);

export default Subject;
